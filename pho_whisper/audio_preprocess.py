import numpy as np
import soundfile as sf
import torchaudio
import noisereduce as nr
import webrtcvad
from scipy.signal import butter, lfilter
from config import TARGET_SR, HPF_CUTOFF, PRE_EMPH, AGGRESSIVENESS
import torch

def resample_and_mono(audio, sr):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).float(), sr, TARGET_SR
        ).numpy()
    return audio


def highpass(audio, sr=TARGET_SR, cutoff=HPF_CUTOFF, order=4):
    ny = 0.5 * sr
    b, a = butter(order, cutoff / ny, btype="high")
    return lfilter(b, a, audio)


def pre_emphasis(x, coeff=PRE_EMPH):
    return np.append(x[0], x[1:] - coeff * x[:-1])


def remove_silence_vad(audio, sr=TARGET_SR, aggressiveness=AGGRESSIVENESS):
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    frames = [audio[i:i + frame_len] for i in range(0, len(audio), frame_len)]
    speech_frames = []
    for f in frames:
        if len(f) < frame_len:
            f = np.pad(f, (0, frame_len - len(f)))
        if vad.is_speech((f * 32768).astype(np.int16).tobytes(), sample_rate=sr):
            speech_frames.append(f)
    if not speech_frames:
        return np.zeros(0)
    return np.concatenate(speech_frames)


def preprocess_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    assert len(audio) > 0, "File âm thanh rỗng!"
    audio = resample_and_mono(audio, sr)
    audio = highpass(audio)
    audio = pre_emphasis(audio)
    audio = nr.reduce_noise(y=audio, sr=TARGET_SR)
    audio = remove_silence_vad(audio)
    assert len(audio) > 0, "Không phát hiện giọng nói sau khi cắt im lặng!"
    # Chuẩn hóa RMS
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio * (0.1 / rms)
    return audio
