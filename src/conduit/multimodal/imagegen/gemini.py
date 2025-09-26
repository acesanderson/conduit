from Chain.message.imagemessage import ImageMessage
from openai import OpenAI
import os

gemini_api_key = os.getenv("GOOGLE_API_KEY")

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

response = client.images.generate(
    model="imagen-3.0-generate-002",
    prompt="A vaporwave computer",
    response_format="b64_json",
    n=1,
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
