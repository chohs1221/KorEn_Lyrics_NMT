#%%
import os
import random
import gc
from typing import Dict, List
import csv

from easydict import EasyDict as edict

import wandb

import numpy as np

import torch

from transformers import(
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
)


#%%
args = edict({'w_project': 'test_project',
              'w_entity': 'chohs1221',
              'pretraining': False,
              'lang': 'en2kor',
              'learning_rate': 1e-4,
              'batch_size': {'train': 2,
                             'eval': 4,},
              'accumulate': 64,
              'epochs': 15,
              'seed': 42,
              'model_path': 'facebook/m2m100_418M',
              })

if args.pretraining:
    args['NAME'] = f'm2m100_kor2en_ep{args.epochs}_lr{args.learning_rate}_{random.randrange(100, 1000)}_pre'
else:
    args['NAME'] = f'm2m100_kor2en_ep{args.epochs}_lr{args.learning_rate}_{random.randrange(100, 1000)}_fine'
print(args.NAME)


#%%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(args.seed)


#%%
wandb.login()

wandb.init(project = args.w_project, entity = args.w_entity)
wandb.run.name = args.NAME


#%%
model = M2M100ForConditionalGeneration.from_pretrained(args.model_path)
tokenizer = M2M100Tokenizer.from_pretrained(args.model_path)
if args.lang == 'kor2en':
    tokenizer.src_lang = "ko"
    tokenizer.tgt_lang = "en"
elif args.lang == 'en2kor':
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "ko"


#%%
class PairedDataset:
    def __init__(self, data):
        self.data = data

    @classmethod
    def loads(cls, *file_names):
        data = []
        for file_name in file_names:
            try:
                with open(file_name, 'r', encoding='cp949') as fd:
                    data += [row[1:] for row in csv.reader(fd)]
            except:
                with open(file_name, 'r', encoding='utf-8') as fd:
                    data += [row[1:] for row in csv.reader(fd)]

        return cls(data)
    
    @classmethod
    def split(cls, datasets, ratio = 0.1):
        valid_length = int(len(datasets) * ratio)
        train = [datasets[i] for i in range(len(datasets) - valid_length)]
        valid = [datasets[i] for i in range(valid_length, len(datasets))]

        return cls(train), cls(valid)

    def __getitem__(self, index: int) -> List[str]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


#%%
if args.pretraining:
    dataset = PairedDataset.loads('./data/train.csv', './data/dev.csv')
else:
    dataset = PairedDataset.loads('./data/kor2en_train.csv', './data/bt_data.csv')
train_dataset_, valid_dataset_ = PairedDataset.split(dataset)
# print(train_dataset_[0])
# print(valid_dataset_[0])


#%%
class TokenizeDataset:
    def __init__(self, dataset, tokenizer, lang):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.lang = lang
    
    def __getitem__(self, index: int):
        if self.lang == 'kor2en':
            src, trg = self.dataset[index]
        elif self.lang == 'en2kor':
            trg, src = self.dataset[index]

        input = self.tokenizer(src, return_attention_mask=False, return_token_type_ids=False, truncation = True, max_length = 512)
        with tokenizer.as_target_tokenizer():
            input['labels'] = self.tokenizer(trg, return_attention_mask=False)['input_ids']

        return input
    
    def __len__(self):
        return len(self.dataset)


#%%
train_dataset = TokenizeDataset(train_dataset_, tokenizer, args.lang)
valid_dataset = TokenizeDataset(valid_dataset_, tokenizer, args.lang)
# print(train_dataset[0])
# print(enc_tokenizer.convert_ids_to_tokens(train_dataset[0]['input_ids']))
# print(dec_tokenizer.convert_ids_to_tokens(train_dataset[0]['labels']))


# %%
collator = DataCollatorForSeq2Seq(tokenizer, model)

arguments = Seq2SeqTrainingArguments(
    output_dir='checkpoints',
    do_train=True,
    do_eval=True,

    num_train_epochs=args.epochs,
    learning_rate = args.learning_rate,
    warmup_ratio=0.1,

    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,

    per_device_train_batch_size=args.batch_size.train,
    per_device_eval_batch_size=args.batch_size.eval,
    gradient_accumulation_steps=args.accumulate,
    dataloader_num_workers=1,
    fp16=True,

    report_to='wandb',
    run_name='test2'
)

trainer = Trainer(
    model,
    arguments,
    data_collator=collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)


#%%
gc.collect()
torch.cuda.empty_cache()


# %%
trainer.train()
model.save_pretrained(f"checkpoints/{args.NAME}")


#%%
wandb.finish()


#%%
model = M2M100ForConditionalGeneration.from_pretrained(f'./checkpoints/{args.NAME}')

input_prompt  = 'let it go let it go'
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
print(100 * '=' + "\nInput:")
print(input_prompt)
outputs = model.generate(input_ids,
                        num_beams=5,
                        num_return_sequences=5,
                        max_length=50,
                        no_repeat_ngram_size = 2,
                        forced_bos_token_id=tokenizer.get_lang_id("ko"))
print(50 * '- ' + "\nOutput:")
for i, output in enumerate(outputs):
  print("{}: {}".format(i, tokenizer.batch_decode(output, skip_special_tokens=True)))
print(100*'=')