from __future__ import annotations
import argparse
from conduit.core.model.model_sync import ModelSync

# DEFAULT_MODEL = "imagegen"
DEFAULT_MODEL = "gemini-2.5-flash-image"


def main():
    parser = argparse.ArgumentParser(description="Generate an image from a prompt.")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    args = parser.parse_args()

    model = ModelSync(model=args.model)
    response = model.image.generate(prompt_str=args.prompt)
    response.display()


if __name__ == "__main__":
    main()
