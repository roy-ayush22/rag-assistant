# RAG Document Assistant

A PDF question answering prototype built from the retrieval and generation steps in `notebook/document.ipynb`. FastAPI serves the RAG pipeline, and the React + TypeScript frontend displays answers with their retrieved source pages.

## How it works

1. PDF pages in `data/` are split into 1000-character chunks with 500-character overlap.
2. `all-MiniLM-L6-v2` embeds each chunk and question.
3. ChromaDB searches the persistent `data/vector_store` collection.
4. Groq generates a grounded answer and cites the retrieved source numbers.

The API reuses the existing `pdf_documents` collection. If the collection is empty, it indexes PDFs from `data/` on startup. The embedding model is loaded locally; the first run may download it.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set your Groq API key. The existing notebook uses `GROQ_API`; `GROQ_API_KEY` is also accepted.

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Run locally

Start the API from the project root:

```bash
uvicorn app.main:app --reload
```

In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open the Vite URL printed in the terminal (usually `http://localhost:5173`). Vite forwards `/api` requests to FastAPI at `http://127.0.0.1:8000`. API docs are available at `http://127.0.0.1:8000/docs`.

## API

`POST /ask`

```json
{
  "question": "What is a synchronous generator?",
  "top_k": 5
}
```

The response contains an `answer` and a `sources` array. Each source includes the PDF filename, one-based page number, retrieved text, and cosine similarity.

`GET /health` reports the number of indexed chunks.
