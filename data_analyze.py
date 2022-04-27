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
    PreTrainedTokenizerFast as BaseGPT2Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
    DistilBertTokenizer,
    
)


#%%
args = edict({'w_project': 'test_project',
              'w_entity': 'chohs1221',
              'pretraining': False,
              'learning_rate': 1e-4,
              'batch_size': {'train': 8,
                             'eval': 4,},
              'accumulate': 32,
              'epochs': 15,
              'seed': 42,
              'model_path': {'encoder': 'distilbert-base-uncased',
                            'decoder': 'skt/kogpt2-base-v2'},
              })


#%%
class PreTrainedTokenizerFast(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]


#%%
enc_tokenizer = DistilBertTokenizer.from_pretrained(args.model_path.encoder)
dec_tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path.decoder, bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')


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

    def __getitem__(self, index: int) -> List[str]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


dataset = PairedDataset.loads('data/kor2en_all.csv')


#%%
class TokenizeDataset:
    def __init__(self, dataset, enc_tokenizer, dec_tokenizer):
        self.dataset = dataset
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
    
    def __getitem__(self, index: int):
        trg, src = self.dataset[index]
        input = self.enc_tokenizer(src, return_attention_mask=False, return_token_type_ids=False)
        
        return input['input_ids']
    
    def __len__(self):
        return len(self.dataset)


#%%
dataset = TokenizeDataset(dataset, enc_tokenizer, dec_tokenizer)


#%%
input_length = []
over512 = []
for i in range(len(dataset)):
    input_length.append(len(dataset[i]))
print(max(input_length))
print(min(input_length))
print(sum(input_length) / len(input_length))
for idx, i in enumerate(input_length):
    if i > 512:
        over512.append((i, idx))
print(over512)