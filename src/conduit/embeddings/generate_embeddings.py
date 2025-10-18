from conduit.embeddings.chroma_batch import ChromaBatch
from pathlib import Path
import json

_DEVICE_CACHE = None
EMBEDDING_MODELS_FILE = Path(__file__).parent / "embedding_models.json"


def get_embedding_models() -> list[str]:
    embedding_models_dict = json.loads(EMBEDDING_MODELS_FILE.read_text())
    embedding_models: list[str] = embedding_models_dict["embedding_models"]
    return embedding_models


def detect_device():
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        import torch

        _DEVICE_CACHE = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    return _DEVICE_CACHE


def generate_embeddings(batch: ChromaBatch, model_name: str) -> ChromaBatch:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = detect_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    inputs = tokenizer(
        batch.documents, padding=True, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()

    batch.embeddings = embeddings
    return batch


if __name__ == "__main__":
    print("Available embedding models:")
    for model in get_embedding_models():
        print(f"- {model}")

    device = detect_device()
    print(f"Detected device: {device}")

    example_batch = ChromaBatch(
        ids=["1", "2"],
        documents=["Hello world", "Goodbye world"],
    )
    print(example_batch.model_dump_json(indent=2))
    model_name = get_embedding_models()[0]
    enriched_batch = generate_embeddings(example_batch, model_name)
    print(enriched_batch.model_dump_json(indent=2))
