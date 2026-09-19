import numpy as np


def retrieve_chunks(
    question_embedding,
    vector_store,
    chunks,
    k=3,
    max_distance=None
):
    """
    Retrieve the top-k relevant chunks for a question.

    Improvements over the earlier version:
    - True per-PDF distribution: each distinct source PDF gets a fair
      share of the k slots (instead of only guaranteeing "at least one
      chunk per source" before falling back to whatever's next-best).
    - Optional relevance threshold: chunks whose L2 distance exceeds
      max_distance are excluded entirely. Left as None by default so
      existing behavior doesn't change unless you opt in.
    - FAISS's -1 "no result" placeholder indices are filtered out
      (fixes a bug where they could silently wrap around to the last
      chunk in the list).
    """

    if not chunks or vector_store.ntotal == 0:
        return []

    search_k = min(len(chunks), max(k * 5, 20))

    distances, indices = vector_store.search(
        np.array([question_embedding]).astype("float32"),
        search_k
    )

    # Build a clean candidate list: drop -1 placeholders and anything
    # past the relevance threshold (if one is set). FAISS already
    # returns these in ascending distance order (best match first).
    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        if max_distance is not None and dist > max_distance:
            continue
        candidates.append((idx, dist))

    if not candidates:
        return []

    # Group candidates by source PDF, preserving relevance order
    # within each group.
    by_source = {}
    for idx, dist in candidates:
        source = chunks[idx]["source"]
        by_source.setdefault(source, []).append((idx, dist))

    sources = list(by_source.keys())
    num_sources = len(sources)

    per_source_quota = max(1, k // num_sources) if num_sources else k

    selected = []
    used_indices = set()

    # First pass: take each source's best chunks, up to its quota.
    for source in sources:
        for idx, dist in by_source[source][:per_source_quota]:
            selected.append((idx, dist))
            used_indices.add(idx)

    # Second pass: if quotas didn't fill k (e.g. k doesn't divide
    # evenly across sources), fill remaining slots with the next-best
    # matches overall, regardless of source.
    if len(selected) < k:
        remaining = [
            (idx, dist) for idx, dist in candidates
            if idx not in used_indices
        ]
        remaining.sort(key=lambda pair: pair[1])

        for idx, dist in remaining:
            if len(selected) >= k:
                break
            selected.append((idx, dist))
            used_indices.add(idx)

    # Final ordering: best match first, capped at k.
    selected.sort(key=lambda pair: pair[1])
    selected = selected[:k]

    relevant_chunks = []
    for idx, dist in selected:
        enriched = dict(chunks[idx])
        enriched["score"] = float(dist)
        relevant_chunks.append(enriched)

    return relevant_chunks
