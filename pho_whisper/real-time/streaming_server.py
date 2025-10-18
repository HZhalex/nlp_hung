import warnings
warnings.filterwarnings("ignore")

import asyncio
import websockets
import numpy as np
import signal
import sys

from audio_preprocess_v2 import preprocess_audio_chunk
from asr_inference_v2 import load_model, transcribe_chunk
from text_postprocess import postprocess_text

print("Đang load model")
model, processor = load_model()
print("Model đã load.")

async def asr_server(websocket):   # chỉ 1 tham số
    print("Client connected:", websocket.remote_address)
    try:
        async for message in websocket:
            # parse bytes -> float32 or int16 fallback
            try:
                chunk = np.frombuffer(message, dtype=np.float32)
            except Exception:
                try:
                    chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                except Exception:
                    await websocket.send("ERROR: unsupported audio format")
                    continue

            try:
                chunk = preprocess_audio_chunk(chunk)
                text = transcribe_chunk(chunk, model, processor)
                text = postprocess_text(text)
            except Exception as e:
                # log và gửi thông báo lỗi ngắn cho client (không crash server)
                print("Error during processing:", e)
                await websocket.send("ERROR: internal processing error")
                continue

            await websocket.send(text)
    except websockets.ConnectionClosedOK:
        print("Client disconnected (normal).")
    except websockets.ConnectionClosedError as e:
        print("Client disconnected (error):", e)
    except Exception as e:
        # report unexpected server-side error; try gửi message ngắn trước đóng
        print("Unhandled error in asr_server:", e)
        try:
            await websocket.send("ERROR: server internal error")
            await websocket.close(code=1011, reason="server error")
        except Exception:
            pass

async def main(host: str = "localhost", port: int = 8765):
    async with websockets.serve(asr_server, host, port):
        print(f"Streaming server đang chạy trên ws://{host}:{port}")
        stop = asyncio.Future()

        def _signal_handler():
            if not stop.done():
                stop.set_result(None)

        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass

        await stop

if __name__ == "__main__":
    try:
        asyncio.run(main("localhost", 8765))
    except KeyboardInterrupt:
        print("Server stopped by user.")
        sys.exit(0)
