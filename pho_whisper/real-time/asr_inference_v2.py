import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from config_v2 import MODEL_ID, TARGET_SR, CHUNK_DURATION

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(device)
    return model, processor


def transcribe_audio(audio, model, processor):
    total_len = len(audio)
    step = int(CHUNK_DURATION * TARGET_SR)
    texts = []
    for start in range(0, total_len, step):
        chunk = audio[start:start + step]
        inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            ids = model.generate(**inputs)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        texts.append(text.strip())
    return " ".join(texts)
def transcribe_chunk(chunk, model, processor):
    if len(chunk) == 0:
        return ""
    inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        ids = model.generate(**inputs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text.strip()

