import warnings
warnings.filterwarnings("ignore")
import asyncio
import websockets
import numpy as np
import sounddevice as sd
from config_v2 import TARGET_SR, STREAM_CHUNK_SEC

async def read_chunk(stream, chunk_len):
    # chạy đọc blocking trong thread worker
    return await asyncio.to_thread(stream.read, chunk_len)

async def stream_mic():
    chunk_len = int(STREAM_CHUNK_SEC * TARGET_SR)

    try:
        async with websockets.connect("ws://localhost:8765") as ws:
            print("Bắt đầu nói")
            with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype="float32") as stream:
                while True:
                    audio_chunk, _ = await read_chunk(stream, chunk_len)
                    audio_chunk = audio_chunk[:, 0]  # (N,1) -> (N,)
                    await ws.send(audio_chunk.tobytes())
                    try:
                        text = await ws.recv()
                    except websockets.ConnectionClosed as e:
                        print("Connection closed by server:", e)
                        break
                    print(f"text: {text}")
    except ConnectionRefusedError:
        print("Không kết nối được tới server. Hãy đảm bảo server đang chạy.")
    except Exception as e:
        print("Lỗi client:", e)

if __name__ == "__main__":
    asyncio.run(stream_mic())
