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
class GPT2Tokenizer(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]


#%%
enc_tokenizer = KoBertTokenizer.from_pretrained(args.model_path.encoder)
dec_tokenizer = GPT2Tokenizer.from_pretrained(args.model_path.decoder)


#%%
model = EncoderDecoderModel.from_pretrained('./checkpoints/best_model')
model.config.decoder_start_token_id = dec_tokenizer.bos_token_id

input_ids = ''
print("Input:\n" + 100 * '-')
print(input_ids)
outputs = model.generate(torch.tensor([enc_tokenizer.encode(input_ids)]),
                        num_beams=10,
                        num_return_sequences=10,)
print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs):
  print("{}: {}".format(i, dec_tokenizer.decode(output, skip_special_tokens=True)))