"""
Eleven Multilingual v2 (eleven_multilingual_v2)
- Excels in stability, language diversity, and accent accuracy
- Supports 29 languages
- Recommended for most use cases

Eleven Flash v2.5 (eleven_flash_v2_5)
- Ultra-low latency
- Supports 32 languages
- Faster model, 50% lower price per character

Eleven Turbo v2.5 (eleven_turbo_v2_5)
- Good balance of quality and latency
- Ideal for developer use cases where speed is crucial
- Supports 32 languages
"""

def tts_elevenlabs(text: str) -> bytes:
    from elevenlabs.client import ElevenLabs
    import os
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    client = ElevenLabs(api_key=api_key)
    
    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    
    # Collect all chunks and concatenate as raw bytes
    # (ElevenLabs streaming chunks are meant to be joined as raw data)
    mp3_chunks = list(audio)
    combined_mp3_bytes = b''.join(mp3_chunks)
    
    return combined_mp3_bytes
