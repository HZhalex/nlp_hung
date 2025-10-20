from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
import soundfile as sf
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from contextlib import asynccontextmanager
import asyncio
import os
import logging
MODEL_NAME = "vinai/PhoWhisper-base"

processor = None
model = None
device = None

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[lifespan] Starting app. Device = {device}")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        # Nếu muốn giảm VRAM (GPU hỗ trợ): uncomment
        # model.half()
        logger.info("[lifespan] Model loaded.")
    except Exception as e:
        logger.exception("[lifespan] Error loading model")
        # Thường muốn container fail fast để biết config sai
        raise

    try:
        yield
    finally:
        logger.info("[lifespan] Shutting down, freeing model.")
        try:
            del model
            del processor
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

app = FastAPI(lifespan=lifespan)

def _read_audio_to_mono_float32(path: str):
    audio, sr = sf.read(path, dtype="float32")
    # Nếu stereo -> convert to mono by averaging channels
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    global processor, model, device
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    suffix = "." + file.filename.split(".")[-1] if "." in file.filename else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            tmp_path = tmp.name

        audio, sr = _read_audio_to_mono_float32(tmp_path)

        # Prepare inputs (processor returns numpy->tensors)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        # Move tensors to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run generation in threadpool to avoid blocking event loop
        generated_ids = await asyncio.to_thread(model.generate, **inputs)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": transcription}

    except Exception as e:
        logger.exception("Error during transcription")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# Lưu ý: không chạy server trực tiếp ở đây trong container.
# Dùng uvicorn từ command line: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1