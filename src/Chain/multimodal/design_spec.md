# Design Specification: Image Generation and TTS Extensions

## Overview

This specification extends the Chain framework to support image generation and text-to-speech (TTS) capabilities through the existing client architecture. The implementation maintains consistency with current patterns while adding multimodal generation support for OpenAI, Google, and HuggingFace providers.

## Core Design Principles

[x] 1. **Extend existing clients** rather than creating separate multimodal clients
[x] 2. **Leverage existing Request/Response flow** with new request types
[x] 3. **Maintain provider-agnostic interface** through Request class orchestration
[x] 4. **Auto-generate appropriate Message objects** (ImageMessage/AudioMessage)
[x] 5. **Preserve caching and serialization** capabilities

## CLI tools
- `tts`
- `imagegen`

## Implementation Requirements

### 1. Request Class Extensions

#### New Fields
```python
class Request(BaseModel, ...):
    # NEW: Request type discrimination
    output_type: Literal["text", "image", "audio"]
```
#### Updated Format Methods
Extend existing `to_openai()`, `to_google()`, `to_ollama()` methods to handle generation request types with provider-specific parameter mapping.

### 2. ClientParams Extensions

#### OpenAIParams
```python
class OpenAIParams(ClientParams):
    # Existing text parameters...
    
    # Image generation parameters
    image_size: Optional[str] = None  # e.g., "1024x1024", "1792x1024"
    image_quality: Optional[str] = None  # "standard", "hd" 
    image_style: Optional[str] = None  # "vivid", "natural"
    image_n: Optional[int] = None  # Number of images
    
    # TTS parameters
    voice: Optional[str] = None  # e.g., "alloy", "echo", "fable"
    tts_model: Optional[str] = None  # "tts-1", "tts-1-hd"
    speed: Optional[float] = None  # 0.25 to 4.0
```

#### GoogleParams  
```python
class GoogleParams(OpenAIParams):
    # Inherits OpenAI params for compatibility
    
    # Google-specific TTS parameters
    voice_name: Optional[str] = None  # e.g., "Kore", "Aoede"
    # (Google image generation parameters TBD based on actual API)
```

#### HuggingFaceParams
```python
class OllamaParams(OpenAIParams):
    # Inherits OpenAI params for compatibility
    # (Local model parameters TBD based on actual capabilities)
```

### 3. Client Method Extensions

#### Implementation Pattern
[ ] - Validate model capabilities vs. modelspec
[ ] - Convert request to provider-specific format
[ ] - Call provider API
[x] - Create appropriate Message object with generated content
[x] - Return `(Message, Usage)` tuple

### 4. Model Class Integration

#### Primary Interface (Option B)
```python
# Through existing query method with Request objects
model = Model("dall-e-3")
request = Request.for_image_generation(
    model="dall-e-3", 
    prompt="a red sports car",
    client_params={"image_size": "1024x1024", "image_quality": "hd"}
)
response = model.query(request=request)
# response.content contains ImageMessage
```

#### Future Enhancement (TBD)
```python
# Currently, you set "output_type" to either "audio" or "image" for tts/imagegen (query_input string is the prompt, naturally)
model = Model("flux1")
response = model.query(query_input = "a red car", output_type = "image")
response.message.display()

# Convenience methods on Model class
## .generate_image
model = Model("dall-e-3")
response = model.generate_image("a red car", size="1024x1024")
response.message.display()

## .generate_audio
model = Model("openai/whisper-base")
response = model.generate_audio("i am the very model of a modern major general")
response.message.play()

# Convenience methods on Response class
response.play()
response.display()
```


### 5. Provider-Specific Implementation Notes

#### OpenAI
- **Image Models**: DALL-E 3, DALL-E 2
- **TTS Models**: tts-1, tts-1-hd  
- **API Endpoints**: `/v1/images/generations`, `/v1/audio/speech`

#### Google  
- **TTS Models**: gemini-2.5-flash-preview-tts
- **Image Models**: (TBD - likely through Gemini models)
- **Integration**: Through existing Google client using OpenAI SDK compatibility

#### HuggingFace
- **Models**: openai/whisper-base, others
- **Integration**: Through HuggingFaceClient which will be an interesting mess of custom pipelines

#### Others
- **Find on GitHub**

### 6. Response Integration

The existing `Response` class accommodates generated content through its support for Message objects. Generated ImageMessage/AudioMessage objects will be returned in the response content, maintaining consistency with current text response handling.

### 7. Caching Integration

No changes required to caching system. Generated content will be cached through existing mechanisms:
- Request cache keys include generation parameters
- ImageMessage/AudioMessage serialization already implemented
- Cache validation works through existing hash comparison

## Validation & Error Handling

- Validate generation capabilities against ModelSpec when available
- Graceful fallback when generation not supported by provider
- Maintain existing ChainError patterns for generation failures
- Provider-specific parameter validation through existing ClientParams system

This specification maintains architectural consistency while cleanly extending multimodal capabilities through the existing request/response flow.
