import csv
from tqdm import tqdm

from typing import List

from easydict import EasyDict as edict

from nltk.translate.bleu_score import sentence_bleu

from lib.tokenization_kobert import KoBertTokenizer

from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    EncoderDecoderModel,
    PreTrainedTokenizerFast as BaseKoGPT2Tokenizer,
    GPT2Tokenizer as BaseGPT2Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
    DistilBertTokenizer,
    
)

from googletrans import Translator


file_name = 'data/kor2en_test.csv'
try:
    with open(file_name, 'r', encoding='cp949') as fd:
        datas = [row[1:] for row in csv.reader(fd)]
except:
    with open(file_name, 'r', encoding='utf-8') as fd:
        datas = [row[1:] for row in csv.reader(fd)]

test_datasets = []
references = []
for data in datas:
    if len(data[1]) < 512:
        test_datasets.append(data[1])
        references.append(data[0])
references = [[i.split()] for i in references]

class PreTrainedTokenizerFast(BaseKoGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]

enc_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dec_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

model = EncoderDecoderModel.from_pretrained(f'./checkpoints/bert_kogpt2_ep15_lr0.0001_692_fine2')

predictions = []
for input_prompt in tqdm(test_datasets):
    input_ids = enc_tokenizer.encode(input_prompt, return_tensors='pt')

    outputs = model.generate(input_ids,
                            # num_beams=5,
                            # num_return_sequences=1,
                            max_length=512,
                            no_repeat_ngram_size = 2)
    
    predictions.append(dec_tokenizer.decode(outputs[0], skip_special_tokens=True).split())

score = 0
cnt = 0
for i in range(len(predictions)):
    cnt += 1
    temp = sentence_bleu(references[i], predictions[i], weights=(0.25, 0.25, 0.25, 0.25))
    # print(temp)
    score += temp
score /= cnt
print('-'*100)
print(f'Bleu Score: {score}')
print('-'*100)

translator = Translator()

google_predictions = []
for input_prompt in tqdm(test_datasets):
    result = translator.translate(input_prompt, src='en', dest="ko")
    google_predictions.append(result.text.split())

score = 0
cnt = 0
for i in range(len(google_predictions)):
    cnt += 1
    temp = sentence_bleu(references[i], google_predictions[i],weights=(0.25, 0.25, 0.25, 0.25))
    # print(temp)
    score += temp
score /= cnt
print('-'*100)
print(f'Google Bleu Score: {score}')
print('-'*100)

with open(f'prediction_bert_kogpt2.csv', 'w') as fd:
    writer = csv.writer(fd)

    writer.writerows(predictions)

with open(f'g_prediction_bert_kogpt2.csv', 'w') as fd:
    writer = csv.writer(fd)

    writer.writerows(google_predictions)





    