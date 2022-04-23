from flask import Flask, request
app = Flask(__name__)

from threading import Semaphore

from transformers import (
    EncoderDecoderModel,
    GPT2Tokenizer,
)

from lib.tokenization_kobert import KoBertTokenizer

src_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
trg_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

model = EncoderDecoderModel.from_pretrained('dump/best_model')
model.config.decoder_start_token_id = trg_tokenizer.bos_token_id
model.eval()
model.cuda()

semaphore = Semaphore(5)

@app.route("/")
def home():
    return "테스트 중 입니다."

@app.route("/ko-en")
def translate():
    text = request.args.get("text")
    print(text)
    embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    with semaphore:
        embeddings = {k: v.cuda() for k, v in embeddings.items()}
        output = model.generate(**embeddings)[0, 1:-1].cpu()
        del embeddings
    return trg_tokenizer.decode(output)

if __name__ == "__main__":
    app.run()