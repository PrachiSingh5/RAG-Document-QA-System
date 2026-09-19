try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(page_texts, chunk_size=500, chunk_overlap=100, min_chunk_length=1):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_id = 0

    for page_data in page_texts:

        page_number = page_data["page"]
        page_text = page_data["text"]
        source = page_data.get("source", "Unknown PDF")

        split_chunks = splitter.split_text(page_text)

        for chunk_text in split_chunks:

            cleaned = chunk_text.strip()

            # Skip near-empty fragments (stray page numbers,
            # whitespace, form-feed artifacts, etc.)
            if len(cleaned) < min_chunk_length:
                continue

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "page": page_number,
                    "text": cleaned
                }
            )

            chunk_id += 1

    return chunks