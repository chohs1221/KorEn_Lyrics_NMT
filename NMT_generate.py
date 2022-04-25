#%%
import random
from typing import List

from easydict import EasyDict as edict

from lib.tokenization_kobert import KoBertTokenizer
from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer as BaseGPT2Tokenizer,
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
model = EncoderDecoderModel.from_pretrained(f'./checkpoints/bertgpt2_ep20_lr5e-05_967')

input_prompt  = '가지마 가지마 가지마 나는 너를 위해 바보처럼 살아 가는데 너는 어디에 니가 필요한데'
input_ids = enc_tokenizer.encode(input_prompt, return_tensors='pt')
print(100 * '=' + "\nInput:")
print(input_prompt)
outputs = model.generate(input_ids,
                        num_beams=5,
                        num_return_sequences=5,
                        max_length=50,
                        no_repeat_ngram_size = 2)
print(50 * '- ' + "\nOutput:")
for i, output in enumerate(outputs):
  print("{}: {}".format(i, dec_tokenizer.decode(output, skip_special_tokens=True)))
print(100*'=')