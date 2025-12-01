import os
from io import BytesIO
import google.generativeai as genai
from PIL import Image
import argparse
from conduit.domain.message.imagemessage import ImageMessage
import base64


def generate_image(prompt: str) -> ImageMessage:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # model = genai.GenerativeModel("gemini-2.5-flash-image")
    # image = model.generate_content(prompt)
    # save_image_from_response(image, "generated_image.png")
    # text_response, image_response = image.parts
    # text_response = text_response.text

    model = genai.GenerativeModel("gemini-3-pro-image-preview")

    image = model.generate_content(prompt)
    save_image_from_response(image, "generated_image.png")

    image_response = image.parts
    text_response = ""

    # convert image_response.inline_data.data from bytes to base64 string
    base64_image_data = base64.b64encode(image_response.inline_data.data).decode(
        "utf-8"
    )
    image_message = ImageMessage(
        role="assistant",
        text_content=text_response,
        image_content=base64_image_data,
    )
    return image_message


def save_image_from_response(response, filename):
    for c in getattr(response, "candidates", []):
        for p in getattr(c.content, "parts", []):
            if getattr(p, "inline_data", None):
                img = Image.open(BytesIO(p.inline_data.data))
                img.save(filename)
                return filename
    raise RuntimeError("No image returned")


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a prompt.")
    parser.add_argument(
        "prompt", type=str, help="The prompt to generate the image from."
    )
    args = parser.parse_args()
    image_msg = generate_image(args.prompt)
    image_msg.display()


if __name__ == "__main__":
    main()
