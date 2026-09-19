import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Build a FAISS L2 index from a 2D array of embeddings.
    Assumes embeddings are normalized (unit length) upstream in
    create_embeddings(), so L2 distance ranking matches cosine
    similarity ranking.
    """

    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            "create_vector_store() received no embeddings to index. "
            "Check that your PDFs produced readable text and chunks."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def save_vector_store(index, path):
    """Persist a FAISS index to disk (for future reuse across sessions)."""
    faiss.write_index(index, path)


def load_vector_store(path):
    """Load a previously saved FAISS index from disk."""
    return faiss.read_index(path)