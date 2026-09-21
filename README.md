# 📄 RAG Document Question Answering System

An AI-powered assistant that lets users upload PDFs (including scanned
documents and images) and ask natural-language questions about their
content, with cited sources and page numbers.

🔗 **Live app:** https://rag-document-app-system-eg9ungzfnxdj9dtppngcge.streamlit.app/

## Features

- **Multi-PDF upload** with per-file processing progress
- **Direct image upload** — upload a photo/screenshot and ask questions about it
- **OCR support** for scanned PDFs (no embedded text layer) using Tesseract
- **Diagram/image understanding** — optionally describes charts, diagrams,
  and photos embedded in PDF pages using a vision-capable LLM, so you can
  ask things like *"what does the diagram on page 5 show?"*
- **Retrieval-Augmented Generation (RAG)** pipeline: chunking → embeddings
  → FAISS vector search → LLM-generated answers grounded in the uploaded
  documents (with per-PDF fair retrieval, not dominated by a single source)
- **Cited sources**: every answer shows which PDF and page it came from,
  with an expandable excerpt of the original text
- **User accounts**: signup/login with hashed passwords (SQLite + bcrypt)
- **Persistent chat history** per user, viewable and deletable from the sidebar
- **Conversation memory** within a session (follow-up questions like "what
  about him?" are understood using recent chat history)
- **Deployed publicly** on Streamlit Community Cloud, with system
  dependencies (Tesseract OCR) configured via `packages.txt` and secrets
  (Groq API key) managed through Streamlit's encrypted Secrets panel

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Text extraction | pypdf |
| OCR | Tesseract + PyMuPDF (page rendering) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS |
| LLM (answers) | Groq (`openai/gpt-oss-120b`) |
| LLM (vision/diagrams) | Groq (`qwen/qwen3.8-27b`) |
| Database | SQLite |
| Password hashing | bcrypt |
| Deployment | Streamlit Community Cloud |
| Version control | Git + GitHub |

## Project Structure

```
RAG-Document-QA-System/
│
├── app.py                  # Main Streamlit app
├── .env                     # Contains GROQ_API_KEY (not committed)
├── requirements.txt          # Python dependencies
├── packages.txt               # System dependencies (Tesseract, for deployment)
│
├── modules/
│   ├── pdf_loader.py         # Extracts text + page numbers from PDFs
│   ├── ocr.py                 # OCR fallback for scanned PDFs + page rendering
│   ├── vision.py               # Image/diagram description via Groq Vision
│   ├── chunker.py               # Splits extracted text into chunks
│   ├── embeddings.py             # Creates sentence embeddings
│   ├── vector_store.py            # Builds/queries the FAISS index
│   ├── retriever.py                # Retrieves relevant chunks (per-PDF fairness)
│   ├── llm.py                       # Generates answers via Groq
│   ├── auth.py                       # Signup/login logic + password hashing
│   ├── auth_ui.py                     # Login/signup screen (swappable UI)
│   └── db.py                           # SQLite: users + chat history
│
├── database/
│   └── app.db                # Auto-created on first run (not committed)
│
├── docs/
│   └── demo_checklist.md    # Suggested order for presenting the project
│
└── screenshots/               # App screenshots for reports/documentation
```

## Setup (running locally)

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   (Windows) and make sure it's on your system PATH.

3. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

5. Create an account from the Sign Up tab, then log in.

## Deployment

This app is deployed on **Streamlit Community Cloud**, connected directly
to this GitHub repository:

1. Code pushed to GitHub (`.env` and `database/app.db` excluded via
   `.gitignore` to keep secrets and user data private)
2. `packages.txt` tells the server to install Tesseract OCR
3. `requirements.txt` pins compatible package versions (including a
   pinned `httpx` version to avoid a known Groq SDK compatibility issue)
4. The Groq API key is provided through Streamlit Cloud's encrypted
   **Secrets** panel, not committed to the repo
5. Every push to the `main` branch automatically triggers a redeploy

## Notes

- Login sessions last for the current browser tab only (no persistent
  "remember me" cookie) — logging out or closing the tab requires
  logging back in.
- Groq's model lineup changes periodically; if `llm.py` or `vision.py`
  report a model error, check https://console.groq.com/docs/models
  for currently available model names.
- Large scanned PDFs (many pages) can take a few minutes to process on
  first upload, since OCR runs page-by-page.

## Future Improvements

- Similarity-threshold filtering to exclude low-relevance chunks from
  retrieval entirely
- Per-document management (delete/rename uploaded documents)
- Reranking retrieved chunks before sending to the LLM
- Persistent vector store storage across sessions