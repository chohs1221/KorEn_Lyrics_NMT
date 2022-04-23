import streamlit as st

from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer,
)

from lib.tokenization_kobert import KoBertTokenizer

if 'tokenizer' not in st.session_state:
    src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    st.session_state.tokenizer = src_tokenizer, trg_tokenizer
else:
    src_tokenizer, trg_tokenizer = st.session_state.tokenizer

@st.cache
def get_model(bos_token_id):
    model = EncoderDecoderModel.from_pretrained('dump/best_model')
    model.config.decoder_start_token_id = bos_token_id
    model.eval()
    #model.cuda()

    return model

model = get_model(trg_tokenizer.bos_token_id)

st.title("한-영 번역기")
st.subheader("한-영 번역기에 오신 것을 환영합니다!")

kor = st.text_area("입력", placeholder="번역할 한국어")

if st.button("번역!", help="해당 한국어 입력을 번역합니다."):
    embeddings = src_tokenizer(kor, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    #embeddings = {k: v.cuda() for k, v in embeddings.items()}
    output = model.generate(**embeddings)[0, 1:-1].cpu()
    st.text_area("출력", value=trg_tokenizer.decode(output), disabled=True)