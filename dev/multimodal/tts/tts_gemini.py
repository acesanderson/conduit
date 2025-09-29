def tts_gemini(text: str) -> bytes:
    from google import genai
    from google.genai import types
    import os
    import io
    from pydub import AudioSegment

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Generate response using the provided text
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=f"Say: {text}",  # Use the actual text parameter
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],  # CRITICAL: Only AUDIO
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore",  # 30 voice options available
                    )
                )
            ),
        ),
    )
    
    # Get the raw audio bytes and MIME type info
    audio_part = response.candidates[0].content.parts[0]
    raw_audio_bytes = audio_part.inline_data.data
    mime_type = audio_part.inline_data.mime_type
    
    # Extract sample rate from MIME type (e.g., 'audio/L16;codec=pcm;rate=24000')
    sample_rate = 24000  # default
    if 'rate=' in mime_type:
        rate_part = mime_type.split('rate=')[1]
        sample_rate = int(rate_part.split(';')[0])  # Handle any additional params after rate
    
    # Create AudioSegment from raw 16-bit PCM data
    audio_segment = AudioSegment(
        raw_audio_bytes,
        frame_rate=sample_rate,  # Use the actual sample rate from MIME type
        sample_width=2,          # 16-bit = 2 bytes per sample (from L16)
        channels=1               # Mono audio (typical for TTS)
    )
    
    # Convert to WAV bytes for consistency with the pipeline
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    
    return wav_buffer.read()
