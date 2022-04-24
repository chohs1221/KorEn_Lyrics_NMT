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

from lib.tokenization_kobert import KoBertTokenizer
from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer as BaseGPT2Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
)


#%%
for name in ('models'):
    os.makedirs(name, exist_ok=True)


#%%
args = edict({'w_project': 'test_project',
              'w_entity': 'chohs1221',
              'learning_rate': 5e-5,
              'batch_size': {'train': 4,
                             'eval': 4,},
              'accumulate': 1,
              'epochs': 10,
              'seed': 42,
              'model_path': {'encoder': 'monologg/kobert',
                            'decoder': 'distilgpt2'},
              })
              
args['NAME'] = ''f'{args.model_path.encoder[-4:]}{args.model_path.decoder[:-4]}_ep{args.epochs}_lr{args.learning_rate}_{random.randrange(100, 1000)}'
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


#%%
class GPT2Tokenizer(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]


#%%
enc_tokenizer = KoBertTokenizer.from_pretrained(args.model_path.encoder)
dec_tokenizer = GPT2Tokenizer.from_pretrained(args.model_path.decoder)


# %%
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    args.model_path.encoder,
    args.model_path.decoder,    
    pad_token_id=dec_tokenizer.bos_token_id
)
model.config.decoder_start_token_id = dec_tokenizer.bos_token_id


#%%
class PairedDataset:
    def __init__(self, data, enc_tokenizer=enc_tokenizer, dec_tokenizer=dec_tokenizer):
        self.data = data

        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

    @classmethod
    def loads(cls, *file_names):
        data = []
        for file_name in file_names:
            with open(file_name, 'r', encoding='cp949') as fd:
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
dataset = PairedDataset.loads('./data/kor2en.csv')
train_dataset_, valid_dataset_ = PairedDataset.split(dataset)
print(train_dataset_[0])
print(valid_dataset_[0])


#%%
class TokenizeDataset:
    def __init__(self, dataset, enc_tokenizer, dec_tokenizer):
        self.dataset = dataset
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
    
    def __getitem__(self, index: int):
        src, trg = self.dataset[index]
        input = self.enc_tokenizer(src, return_attention_mask=False, return_token_type_ids=False, padding='max_length', truncation = True, max_length = 512)
        input['labels'] = self.dec_tokenizer(trg, return_attention_mask=False)['input_ids']

        return input
    
    def __len__(self):
        return len(self.dataset)


#%%
train_dataset = TokenizeDataset(train_dataset_, enc_tokenizer, dec_tokenizer)
valid_dataset = TokenizeDataset(valid_dataset_, enc_tokenizer, dec_tokenizer)
# print(train_dataset[0])
# print(enc_tokenizer.convert_ids_to_tokens(train_dataset[0]['input_ids']))
# print(dec_tokenizer.convert_ids_to_tokens(train_dataset[0]['labels']))


# %%
collator = DataCollatorForSeq2Seq(enc_tokenizer, model)

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

model.save_pretrained("models/best_model")


#%%
wandb.finish()


#%%
model = EncoderDecoderModel.from_pretrained('./models/best_model')
model.config.decoder_start_token_id = dec_tokenizer.bos_token_id

input_ids = '안녕하세요'
print("Input:\n" + 100 * '-')
print(input_ids)
outputs = model.generate(torch.tensor([enc_tokenizer.encode(input_ids)]),
                        num_beams=5,
                        skip_special_tokens=True,
                        num_return_sequences=5,)
print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs):
  print("{}: {}".format(i, dec_tokenizer.decode(output, skip_special_tokens=True)))