def tts_chatterbox(text: str) -> bytes:
    """
    Convert text to speech using ResembleAI's Chatterbox TTS model.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        WAV audio data as bytes
    """
    import torch
    import soundfile as sf
    import io
    import numpy as np
    from chatterbox.tts import ChatterboxTTS
    
    # Determine device (prefer CUDA if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Generate speech with default settings
    # Default: exaggeration=0.5, cfg=0.5 (good for most use cases)
    wav = model.generate(text)
    
    # Convert tensor to numpy array
    if isinstance(wav, torch.Tensor):
        wav_numpy = wav.detach().cpu().numpy()
    else:
        wav_numpy = wav
    
    # Handle different tensor shapes (squeeze if needed)
    if wav_numpy.ndim > 1:
        wav_numpy = wav_numpy.squeeze()
    
    # Convert to WAV bytes using soundfile
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, wav_numpy, model.sr, format='WAV')
    wav_buffer.seek(0)
    
    return wav_buffer.read()
