import io
from pydub import AudioSegment
from typing import Callable
from pathlib import Path

# Standard text for all TTS models
STANDARD_TEXT = """
"Oh no!" Sarah gasped as thunder crashed overhead. The weatherman had predicted sunshine, but now
heavy raindrops were pelting her umbrella. She quickened her pace, her heels clicking rhythmically
on the wet pavement—click, click, click.

Suddenly, she heard a tiny mewing sound. Behind a rusty dumpster, a small kitten sat shivering.
"Well, hello there, little one," she whispered gently, crouching down. The kitten's eyes sparkled
like emeralds.

"Are you hungry?" she asked, pulling out her tuna sandwich. The kitten purred—a soft, rumbling
vibration that made Sarah smile. "There you go, sweetie."

As the storm passed, golden sunlight broke through the clouds. Sarah laughed with joy, realizing
this unexpected detour had brightened her entire day. Sometimes the most beautiful moments come from
life's surprises.

"Come on, little friend," she said cheerfully, "let's find you a warm home." Together, they walked
toward the rainbow stretching across the clearing sky, their footsteps creating a gentle symphony on
the glistening street.
""".strip()

# Main pipeline
def pipeline(tts_function: Callable, text = STANDARD_TEXT):
    # Construct Path for MP3 file
    model_name = tts_function.__name__.replace("tts_", "")
    mp3_path = Path(__file__).parent / "audio_files" / f"{model_name}.mp3"
    # Begin our pipeline
    print(f"Generating speech using {tts_function.__name__}...")
    audio_bytes = tts_function(text)
    print(f"Converting audio to MP3 for {model_name}...")
    mp3_bytes = coerce_to_mp3(audio_bytes)
    print(f"Saving MP3 for {model_name}...")
    save_mp3(mp3_path, mp3_bytes)

# Abstract base class for TTS models
def generate_speech(text: str = STANDARD_TEXT) -> bytes:
    """
    Generate speech from text using the specific TTS model.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio data as bytes (format depends on the specific model implementation)
    """
    # This is a placeholder - each model implementation will override this
    raise NotImplementedError("This method should be implemented by specific TTS model classes")

# Pipeline functions
def coerce_to_mp3(audio_bytes: bytes) -> bytes:
    """
    Convert audio bytes to MP3 format.
    
    Args:
        audio_bytes: Audio data in bytes (WAV, MP3, or other common formats)
        
    Returns:
        MP3 audio data as bytes
    """
    # Load audio from bytes
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    # Convert to MP3
    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)
    
    return mp3_buffer.read()

def save_mp3(filename: Path, mp3_bytes: bytes) -> None:
    """
    Save MP3 bytes to file in working directory.
    
    Args:
        model_name: Name of the TTS model (used as filename)
        mp3_bytes: MP3 audio data as bytes
    """
    filename.write_bytes(mp3_bytes)
    if filename.exists():
        print(f"MP3 file saved successfully: {filename}")
    else:
        print(f"Failed to save MP3 file: {filename}")

# Helper functions that may be needed within generate_speech implementations
def combine_wav_bytes(wav_bytes_list: list) -> bytes:
    """
    Combine multiple WAV bytes objects into a single WAV bytes object.
    
    Args:
        wav_bytes_list: List of WAV audio data as bytes
        
    Returns:
        Combined WAV audio data as bytes
    """
    if not wav_bytes_list:
        raise ValueError("Empty list provided")
    
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]
    
    # Load all audio segments
    audio_segments = []
    for wav_bytes in wav_bytes_list:
        segment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        audio_segments.append(segment)
    
    # Concatenate all segments
    combined = audio_segments[0]
    for segment in audio_segments[1:]:
        combined += segment
    
    # Export back to WAV bytes
    wav_buffer = io.BytesIO()
    combined.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    return wav_buffer.read()

if __name__ == "__main__":
    from tts_elevenlabs import tts_elevenlabs
    # from tts_chatTTS import tts_chatTTS
    from tts_gemini import tts_gemini
    from tts_openai import tts_openai
    # from tts_chatterbox import tts_chatterbox
    tts_functions = [tts_gemini, tts_openai, tts_elevenlabs]
    for tts_function in tts_functions:
        pipeline(tts_function, STANDARD_TEXT)

