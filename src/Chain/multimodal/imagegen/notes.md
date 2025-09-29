Implement image generation for:
- Gemini (requires updating GeminiClient as well as Params)
- OpenAI (requires updating OpenAIClient as well as Params)
- Ollama (requires updating OllamaClient as well as Params)

### OpenAI models
- dall-e 3 (dedicated endpoint)
- gpt-4o, o3-mini, 04-mini (but only through web interface)

tldr: only possible through the dedicated endpoint.

This is the primary and most flexible way to generate images using OpenAI’s API.

How it works:
You send a POST request to the DALL·E endpoint with your text prompt and configuration options.

Supported operations:
Text-to-image generation: Create images from text prompts.
Edits: Edit or extend an existing image (for DALL·E 2; DALL·E 3 currently supports only text-to-image).

Variations: Generate variations of an input image (DALL·E 2).

Example (Python):

```python
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt="A vaporwave computer",
    n=1,
    size="1024x1024"
)

print(response.data[0].url)
```

This returns a URL to the generated image.

Customization:
You can specify parameters like model version, image size, quality, and number of images.


