import argparse


def main():
    parser = argparse.ArgumentParser(description="Tokenize input text.")
    parser.add_argument("text", type=str, help="The text to be tokenized.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt",
        help="The model to use for tokenization.",
    )
    args = parser.parse_args()

    from conduit.sync import Model

    model = Model(args.model)
    tokens: int = model.tokenize(args.text)
    print(f"Number of tokens: {tokens}")


if __name__ == "__main__":
    main()
