import argparse
import sys


def main():
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        parser = argparse.ArgumentParser(description="Tokenize input text.")
        parser.add_argument(
            "--model",
            "-m",
            type=str,
            default="gemini",
            help="The model to use for tokenization.",
        )
        args = parser.parse_args()
        text = stdin_text
    else:
        parser = argparse.ArgumentParser(description="Tokenize input text.")
        parser.add_argument("text", type=str, help="The text to be tokenized.")
        parser.add_argument(
            "--model",
            "-m",
            type=str,
            default="gemini",
            help="The model to use for tokenization.",
        )
        args = parser.parse_args()
        text = args.text

    from conduit.sync import Model

    model = Model(args.model)
    tokens: int = model.tokenize(text)
    print(f"Number of tokens: {tokens}")


if __name__ == "__main__":
    main()
