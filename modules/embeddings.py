import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dimensionality of all-MiniLM-L6-v2

_model = None


def _get_model():
    """Lazily load the embedding model once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def create_embeddings(data, batch_size=32, normalize=True):
    """
    Create embeddings for either:
      - a list of chunk dicts, each with a "text" key, or
      - a list of raw strings (e.g. a single question).

    Always returns a float32 numpy array, ready for FAISS.
    """

    if not data:
        return np.empty((0, EMBEDDING_DIM), dtype="float32")

    if isinstance(data[0], dict):
        texts = [item["text"] for item in data]
    else:
        texts = data

    model = _get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )

    return np.asarray(embeddings, dtype="float32")