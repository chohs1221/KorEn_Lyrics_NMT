import streamlit as st
from typing import Dict, List

from lib.tokenization_kobert import KoBertTokenizer

from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizerFast as KoBaseGPT2Tokenizer,
    GPT2Tokenizer as BaseGPT2Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
    DistilBertTokenizer,
    
)

import gtts


st.set_page_config(page_title='Project3_2jo', page_icon='translator-icon.png', layout='wide', initial_sidebar_state='expanded')
st.title("Lyric Translator:balloon:")

class PreTrainedTokenizerFast(KoBaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]

class GPT2Tokenizer(BaseGPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids: List[int], _) -> List[int]:
        return token_ids + [self.eos_token_id]

if 'en2kor_tokenizer' not in st.session_state:
    en2kor_src_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    en2kor_trg_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    st.session_state.en2kor_tokenizer = en2kor_src_tokenizer, en2kor_trg_tokenizer
else:
    en2kor_src_tokenizer, en2kor_trg_tokenizer = st.session_state.en2kor_tokenizer

if 'kor2en_tokenizer' not in st.session_state:
    kor2en_src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    kor2en_trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    st.session_state.kor2en_tokenizer = kor2en_src_tokenizer, kor2en_trg_tokenizer
else:
    kor2en_src_tokenizer, kor2en_trg_tokenizer = st.session_state.kor2en_tokenizer

@st.cache
def get_en2kor_model():
    model = EncoderDecoderModel.from_pretrained(f'./checkpoints/bert_kogpt2_ep15_lr0.0001_304_fine')
    model.eval()

    return model

@st.cache
def get_kor2en_model():
    model = EncoderDecoderModel.from_pretrained(f'./checkpoints/kobert_gpt2_ep15_lr0.0001_384_fine')
    model.eval()

    return model

en2kor_model = get_en2kor_model()
kor2en_model = get_kor2en_model()


# text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")
text = st.text_area("Enter text:", placeholder="Enter your text here")

option1 = st.selectbox('Input language',
                        ('english' , 'korean'))
option2 = st.selectbox('Output language', 
                        ('korean' , 'english'))
Languages = {'english':'en', 'korean':'ko'}
value1 = Languages[option1]
value2 = Languages[option2]

if st.button('Translate Sentence'):
    if text == "":
        st.warning('Please **enter text** for translation')

    else:
        if value1 == 'en':
            embeddings = en2kor_src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
            output = en2kor_model.generate(**embeddings, max_length=50)[0, 1:-1]
            output=en2kor_trg_tokenizer.decode(output, skip_special_tokens=True)
        elif value1 == 'ko':
            embeddings = kor2en_src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
            output = kor2en_model.generate(**embeddings, max_length=50)[0, 1:-1]
            output=kor2en_trg_tokenizer.decode(output, skip_special_tokens=True)


        st.text_area("Translation", value=output, disabled=True)

        converted_audio = gtts.gTTS(output, lang=value2)
        converted_audio.save("translated.mp3")
        audio_file = open('translated.mp3','rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio')
        st.write("To **download the audio file**, click the kebab menu on the audio bar.")
        st.success("Translation is **successfully** completed!")
        st.balloons()
else:
    pass