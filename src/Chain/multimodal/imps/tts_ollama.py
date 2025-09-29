import requests
from pydub import AudioSegment
from pydub.playback import play

def play_orpheus_audio(audio_data):
    """Play Orpheus audio (24kHz mono WAV)"""
    audio = AudioSegment(data=audio_data, sample_width=2, frame_rate=24000, channels=1)
    play(audio)

def test_orpheus_tts():
    """Test Orpheus TTS with sematre/orpheus:en"""
    url = "http://localhost:11434/api/generate"
    
    prompt = "tara: Hello! This is a test of Orpheus TTS. <laugh>"
    
    payload = {
        "model": "sematre/orpheus:en",
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    print("Response:", result.get('response'))
    
    breakpoint()
    # If audio is returned (hypothetical)
    if "audio" in result:
        import base64
        audio_data = base64.b64decode(result["audio"])
        play_orpheus_audio(audio_data)

if __name__ == "__main__":
    test_orpheus_tts()
