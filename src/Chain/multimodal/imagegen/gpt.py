from Chain.message.imagemessage import ImageMessage
from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
assert api_key

client = OpenAI(api_key=api_key)

response = client.images.generate(
    model="dall-e-3",
    prompt="A vaporwave computer",
    n=1,
    size="1024x1024",
    response_format="b64_json",
)

image_data = response.data[0].b64_json
assert isinstance(image_data, str)


"""
        image_content: str,
        text_content: str,
        mime_type: str = "image/png",
        role: str = "user"
"""


image_message = ImageMessage.from_base64(
    image_content=image_data,
    text_content="A vaporwave computer",
    mime_type="image/png",
    role="user",
)

image_message.display()
