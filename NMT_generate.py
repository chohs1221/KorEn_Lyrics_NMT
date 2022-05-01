#%%
from typing import List

from easydict import EasyDict as edict

from lib.tokenization_kobert import KoBertTokenizer
from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer as BaseGPT2Tokenizer,
    DistilBertTokenizer,
    PreTrainedTokenizerFast as BasekoGPT2Tokenizer,
)


#%%
args = edict({
              'model_path': {'encoder': 'monologg/kobert',
                            'decoder': 'distilgpt2'},
              })

iskor2en = False


#%%
if iskor2en:
    class GPT2Tokenizer(BaseGPT2Tokenizer):
        def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
            return token_ids + [self.eos_token_id]
    enc_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    dec_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = EncoderDecoderModel.from_pretrained(f'./checkpoints/kobert_gpt2_ep15_lr0.0001_384_fine')
else:
    class PreTrainedTokenizerFast(BasekoGPT2Tokenizer):
        def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
            return token_ids + [self.eos_token_id]
    enc_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dec_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    # model = EncoderDecoderModel.from_pretrained(f'./checkpoints/bert_kogpt2_ep10_lr0.0001_611_pre')
    model = EncoderDecoderModel.from_pretrained(f'./checkpoints/bert_kogpt2_ep15_lr0.0001_692_fine2')


#%%

input_prompt  = "I am feelings is this real or just another wishing well you got my temperature arising everytime you hold me in your arms tell me con you feel me burning as you get me closer everytime is this true love that I've been missing is this real or just another passing"
input_ids = enc_tokenizer.encode(input_prompt, return_tensors='pt')
print(100 * '=' + "\nInput:")
print(input_prompt)
outputs = model.generate(input_ids,
                        # num_beams=5,
                        # num_return_sequences=5,
                        max_length=512,
                        no_repeat_ngram_size = 2)
print(50 * '- ' + "\nOutput:")
for i, output in enumerate(outputs):
  print("{}: {}".format(i, dec_tokenizer.decode(output, skip_special_tokens=True)))
print(100*'=')