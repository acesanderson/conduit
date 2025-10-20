import json
from pathlib import Path
from conduit.embeddings.chroma_batch import ChromaBatch
from functools import cached_property

_DEVICE_CACHE = None
EMBEDDING_MODELS_FILE = Path(__file__).parent / "embedding_models.json"


class EmbeddingModel:
    def __init__(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        if model_name not in self.models():
            raise ValueError(
                f"Model '{model_name}' is not in the list of supported models."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device())

    @classmethod
    def models(cls) -> list[str]:
        embedding_models_dict = json.loads(EMBEDDING_MODELS_FILE.read_text())
        embedding_models: list[str] = embedding_models_dict["embedding_models"]
        return embedding_models

    @classmethod
    def device(cls) -> str:
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

    def generate_embeddings(self, batch: ChromaBatch) -> ChromaBatch:
        """
        Generate embeddings for a batch of documents.
        Args:
            batch (ChromaBatch): A batch of ids and documents to generate embeddings for.
        Returns:
            ChromaBatch: A new batch the original ids and documents, as well as generated embeddings.
        """
        import torch

        inputs = self.tokenizer(
            batch.documents, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device()) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()

        new_batch = ChromaBatch(
            ids=batch.ids,
            documents=batch.documents,
            metadatas=batch.metadatas,
            embeddings=embeddings,
        )
        return new_batch


if __name__ == "__main__":
    # Example usage
    model = EmbeddingModel("sentence-transformers/all-mpnet-base-v2")
    batch = ChromaBatch(
        ids=["1", "2"],
        documents=["This is a test document.", "This is another test document."],
        metadatas=[{}, {}],
    )
    embedded_batch = model.generate_embeddings(batch)
    print(embedded_batch.embeddings)
