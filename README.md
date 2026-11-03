# RAG Document Assistant

A PDF question answering prototype built from the retrieval and generation steps in `notebook/document.ipynb`. FastAPI serves the RAG pipeline, and the React + TypeScript frontend displays answers with their retrieved source pages.

## How it works

1. PDF pages in `data/` are split into 1000-character chunks with 150-character overlap.
2. `all-MiniLM-L6-v2` embeds each chunk and question.
3. Qdrant Cloud searches the `electric_motors` collection.
4. Groq generates a grounded answer and cites the retrieved source numbers.

The notebook uploads PDF chunks and their locally generated embeddings to Qdrant. The API connects to that collection and uses Qdrant Cloud Inference with the same MiniLM model to embed each question, so Render does not load the sentence-transformer model. The API does not ingest or index PDFs on startup. Qdrant Cloud Inference must be enabled for the cluster.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and set `GROQ_API`, `QDRANT_URL`, and `QDRANT_API_KEY`. `GROQ_API_KEY` is also accepted. Set `QDRANT_COLLECTION` and `EMBEDDING_MODEL` to the same values for notebook ingestion and API retrieval.

Run the notebook ingestion cells in order through the vector upload cell to send the PDF chunks to Qdrant. The API can answer questions after the collection has been populated.

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
