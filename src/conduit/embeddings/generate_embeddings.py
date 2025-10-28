"""
Generate raw embeddings for a list of texts using a specified embedding model.
"""

from headwater_api.client.headwater_client import HeadwaterClient
from headwater_api.classes import (
    ChromaBatch,
    EmbeddingsRequest,
    EmbeddingsResponse,
    load_embedding_models,
)


def generate_embeddings(
    ids: list[str],
    documents: list[str],
    model: str = "sentence-transformers/all-mpnet-base-v2",
) -> list[list[float]]:
    """
    Get embeddings for a list of texts using the specified model.

    Args:
        texts (list of str): The texts to embed.
        model (str): The embedding model to use.

    Returns:
        list of list of float: The embeddings for the input texts.
    """
    # Validate inputs
    if validate_model(model) is False:
        raise ValueError(f"Model '{model}' is not a valid embedding model.")
    assert len(ids) == len(documents), "Length of ids and documents must match."
    # Prepare request
    chroma_batch = ChromaBatch(
        ids=ids,
        documents=documents,
    )
    request = EmbeddingsRequest(
        model=model,
        batch=chroma_batch,
    )
    # Generate embeddings
    client = HeadwaterClient()
    response: EmbeddingsResponse = client.embeddings.generate_embeddings(request)
    assert len(response.embeddings) == len(documents), (
        "Number of embeddings doesn't match number of documents."
    )
    return response.embeddings


def validate_model(model_name: str) -> bool:
    embedding_models = load_embedding_models()
    return model_name in embedding_models
