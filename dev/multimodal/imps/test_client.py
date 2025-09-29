import numpy as np
from pydub import AudioSegment
from pydub.playback import play

def play_chattts_audio(wav_array):
    audio_int16 = (wav_array * 32767).astype(np.int16)
    audio = AudioSegment(audio_int16.tobytes(), frame_rate=24000, sample_width=2, channels=1)
    play(audio)

def load_and_play():
    loaded = np.load("audio_batch.npz")
    wavs = [loaded[key] for key in loaded.files]
    
    for i, wav in enumerate(wavs):
        print(f"Playing audio {i+1}")
        play_chattts_audio(wav)

if __name__ == "__main__":
    load_and_play()
