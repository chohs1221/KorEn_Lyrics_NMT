import streamlit as st
from typing import Dict, List

from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizerFast as BaseGPT2Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
    DistilBertTokenizer,
    
)

from lib.tokenization_kobert import KoBertTokenizer


class PreTrainedTokenizerFast(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]

if 'tokenizer' not in st.session_state:
    src_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    trg_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    st.session_state.tokenizer = src_tokenizer, trg_tokenizer
else:
    src_tokenizer, trg_tokenizer = st.session_state.tokenizer

@st.cache
def get_model():
    model = EncoderDecoderModel.from_pretrained(f'./checkpoints/bert_kogpt2_ep15_lr0.0001_304_fine')
    model.eval()

    return model

model = get_model()

st.title("한-영 번역기")
st.subheader("한-영 번역기에 오신 것을 환영합니다!")

kor = st.text_area("입력", placeholder="번역할 한국어")

if st.button("번역!", help="해당 영어 입력을 번역합니다."):
    embeddings = src_tokenizer(kor, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    #embeddings = {k: v.cuda() for k, v in embeddings.items()}
    output = model.generate(**embeddings)[0, 1:-1].cpu()
    st.text_area("출력", value=trg_tokenizer.decode(output), disabled=True)