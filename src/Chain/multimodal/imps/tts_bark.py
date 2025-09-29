from bark import SAMPLE_RATE, generate_audio, preload_models
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

def play_bark_audio(audio_array):
    """Play Bark audio array using pydub"""
    # Convert float array to 16-bit PCM
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Create AudioSegment from raw audio
    audio = AudioSegment(
        audio_int16.tobytes(), 
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )
    play(audio)

def test_bark_tts():
    """Test Bark TTS"""
    preload_models()
    
    text_prompt = """
         Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
         But I also have other interests such as playing tic tac toe.
    """
    
    audio_array = generate_audio(text_prompt)
    play_bark_audio(audio_array)

if __name__ == "__main__":
    test_bark_tts()
