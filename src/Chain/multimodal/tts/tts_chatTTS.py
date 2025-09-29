def tts_chatTTS(text: str) -> bytes:
    """
    Convert text to speech using ChatTTS and return the audio data.
    
    Args:
        text (str): The text to convert to speech.
    
    Returns:
        np.ndarray: The audio data as a numpy array.
    """
    import soundfile as sf
    import numpy as np
    import ChatTTS, io
    
    # Initialize and load ChatTTS
    chat = ChatTTS.Chat()
    chat.load(compile=False)  # Set to True for better performance
    
    # Generate speech - returns list of numpy arrays
    wavs = chat.infer(text)
    
    # Concatenate if multiple segments (ChatTTS typically returns one array for simple text)
    if len(wavs) > 1:
        combined_wav = np.concatenate(wavs)
    else:
        combined_wav = wavs[0]
    
    # Convert numpy array to WAV bytes
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, combined_wav, 24000, format='WAV')  # ChatTTS uses 24kHz sample rate
    wav_buffer.seek(0)
    
    return wav_buffer.read()
