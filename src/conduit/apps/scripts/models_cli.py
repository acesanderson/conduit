import argparse
from collections import namedtuple
from conduit.core.model.models.modelstore import ModelStore

models = ModelStore.list_models()
modeltypes = ModelStore.list_model_types()
providers = ModelStore.list_providers()

Match = namedtuple("Match", ["title", "score", "rank"])


def fuzzy_search(query: str, limit: int = 3):
    from rapidfuzz import process, fuzz

    choices = models
    results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
    matches = [
        Match(title=title, score=score, rank=rank + 1)
        for rank, (title, score, _) in enumerate(results)
    ]
    return matches


"""
["BAAI/bge-large-en-v1.5","BAAI/bge-base-en-v1.5","BAAI/bge-reranker-v2-m3","sentence-transformers/all-mpnet-base-v2","sentence-transformers/all-MiniLM-L6-v2","nomic-ai/nomic-embed-text-v1.5","intfloat/e5-large-v2","google/embeddinggemma-300m"]
"""


def main():
    parser = argparse.ArgumentParser(description="CLI for managing models.")
    parser.add_argument(
        "-m", "--model", type=str, help="Name of the model to retrieve details for."
    )
    parser.add_argument(
        "-t", "--type", type=str, help="Type of the model to filter by."
    )
    parser.add_argument(
        "-p", "--provider", type=str, help="Provider of the model to filter by."
    )
    parser.add_argument(
        "-a", "--aliases", action="store_true", help="Display model aliases."
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        action="store_true",
        help="Display models that support embeddings.",
    )
    args = parser.parse_args()
    # Validate arguments
    if args.embeddings:
        from conduit.embeddings.generate_embeddings import list_embedding_models

        embedding_model_str = list_embedding_models()
        # Convert the string representation of the list to an actual list
        embedding_models = embedding_model_str.strip("[]").replace('"', "").split(",")

        from rich.console import Console

        console = Console()
        console.print("Embedding models:", style="bold green")
        for model in embedding_models:
            console.print(f"  - {model}", style="cyan")
        return
    if args.type:
        if args.type not in modeltypes:
            raise ValueError(
                f"Invalid model type: {args.type}. Must be one of: {' | '.join(modeltypes)}."
            )
    if args.provider:
        if args.provider not in providers:
            raise ValueError(
                f"Invalid provider: {args.provider}. Must be one of: {' | '.join(providers)}."
            )
    if args.aliases:
        from rich.console import Console

        console = Console()

        aliases = ModelStore.aliases()
        console.print(aliases)
        return
    if args.model:
        model_string = ModelStore().validate_model
        if not model_string:
            raise ValueError(
                f"Model {args.model} not found. Available models: {', '.join(models)}."
            )
    # Run commands
    if args.model:
        try:
            modelspec = ModelStore.get_model(args.model)
            modelspec.card
        except ValueError:
            matches = fuzzy_search(args.model)
            from rich.console import Console

            console = Console()
            console.print(f"[red]Model '{args.model}' not found. Did you mean:[/red]")
            for match in matches:
                console.print(f"  {match.rank}. {match.title}")
    elif args.type:
        modelspecs = ModelStore.by_type(args.type)
        for model in modelspecs:
            print(model.model)
    elif args.provider:
        modelspecs = ModelStore.by_provider(args.provider)
        for model in modelspecs:
            print(model.model)
    else:
        ModelStore.display()


if __name__ == "__main__":
    main()
