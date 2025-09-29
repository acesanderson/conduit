from conduit.message.imagemessage import ImageMessage
from diffusers import FluxPipeline
import torch, io, base64, argparse


def generate_image(prompt: str) -> ImageMessage:
    pipe = FluxPipeline.from_pretrained(
        "Jlonge4/flux-dev-fp8", torch_dtype=torch.bfloat16
    )
    pipe.enable_sequential_cpu_offload()

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

    # Convert PIL image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # or "JPEG"
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return ImageMessage.from_base64(
        image_content=img_str, text_content="generated image"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a prompt.")
    parser.add_argument(
        "prompt", type=str, help="The prompt to generate the image from."
    )
    args = parser.parse_args()
    im = generate_image(args.prompt)
    im.display()


if __name__ == "__main__":
    main()
