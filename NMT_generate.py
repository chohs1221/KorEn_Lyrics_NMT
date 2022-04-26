#%%
from typing import List

from easydict import EasyDict as edict

from lib.tokenization_kobert import KoBertTokenizer
from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer as BaseGPT2Tokenizer,
)


#%%
args = edict({
              'model_path': {'encoder': 'monologg/kobert',
                            'decoder': 'distilgpt2'},
              })


#%%
class GPT2Tokenizer(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]


#%%
enc_tokenizer = KoBertTokenizer.from_pretrained(args.model_path.encoder)
dec_tokenizer = GPT2Tokenizer.from_pretrained(args.model_path.decoder)


#%%
model = EncoderDecoderModel.from_pretrained(f'./checkpoints/kobert_gpt2_ep15_lr0.0001_384_fine')

input_prompt  = '허세와는 거리가 멀어'
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