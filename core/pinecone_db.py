"""
Pinecone vector database integration.

Manages vector storage and similarity search against a Pinecone index
for the multimodal search engine.

Uses lazy initialization so Django commands like 'migrate' don't fail
when Pinecone credentials aren't configured yet.
"""

import os
from pinecone import Pinecone

# Lazy-initialized globals
_pc = None
_index = None


def _get_index():
    """Lazily initialize and return the Pinecone index connection."""
    global _pc, _index
    if _index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise Exception(
                "PINECONE_API_KEY is not set. "
                "Please add it to your .env file."
            )
        _pc = Pinecone(api_key=api_key)
        index_name = os.environ.get("PINECONE_INDEX", "multimodal-search")
        _index = _pc.Index(index_name)
    return _index


def store_vector(id: str, vector: list, metadata: dict) -> None:
    """
    Store a single vector with metadata in Pinecone.

    Args:
        id: Unique identifier for the vector.
        vector: List of floats representing the embedding.
        metadata: Dictionary of metadata to store alongside the vector.
    """
    _get_index().upsert(
        vectors=[
            {
                "id": id,
                "values": vector,
                "metadata": metadata,
            }
        ]
    )


def search_vectors(query_vector: list, top_k: int = 5, filter_type: str = None) -> list:
    """
    Search for similar vectors in Pinecone.

    Args:
        query_vector: The query embedding vector.
        top_k: Number of top results to return.
        filter_type: Optional filter on the 'type' metadata field.

    Returns:
        A list of match objects from Pinecone with scores and metadata.
    """
    query_params = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if filter_type:
        query_params["filter"] = {"type": {"$eq": filter_type}}

    results = _get_index().query(**query_params)
    return results.matches
