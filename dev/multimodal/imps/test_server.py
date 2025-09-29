# server.py - Generate and save ChatTTS audio
import ChatTTS
import numpy as np

def generate_and_save():
    chat = ChatTTS.Chat()
    chat.load(compile=False)
    
    texts = ["Hello! This is ChatTTS speaking.", "How are you today?"]
    wavs = chat.infer(texts)
    
    np.savez("audio_batch.npz", *wavs)
    print(f"Saved {len(wavs)} audio clips to audio_batch.npz")

if __name__ == "__main__":
    generate_and_save()
