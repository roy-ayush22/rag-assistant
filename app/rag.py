from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "pdf_documents"
load_dotenv(ROOT_DIR / ".env")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class RagService:
    """Loads the notebook's persisted Chroma index and answers user questions."""

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "pdf documents for rag",
                "hnsw:space": "cosine",
            },
        )

        if self.collection.count() == 0:
            self._ingest_pdfs()

    def _ingest_pdfs(self) -> None:
        """Build the same 1000-character chunks used by the notebook if needed."""
        pdfs = list(DATA_DIR.rglob("*.pdf"))
        if not pdfs:
            return

        pages = DirectoryLoader(
            str(DATA_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader
        ).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        if not chunks:
            return

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            source = str(metadata.get("source", "unknown"))
            page = metadata.get("page", 0)
            digest = hashlib.sha256(
                f"{source}:{page}:{index}:{chunk.page_content}".encode("utf-8")
            ).hexdigest()[:24]
            ids.append(digest)
            metadata["doc_index"] = index
            metadata["content_length"] = len(chunk.page_content)
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

    def ask(self, question: str, top_k: int = 5) -> dict[str, Any]:
        query_embedding = self.embedding_model.encode(
            [question], show_progress_bar=False
        )[0]
        matches = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, max(self.collection.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )

        documents = matches["documents"][0] if matches["documents"] else []
        metadatas = matches["metadatas"][0] if matches["metadatas"] else []
        distances = matches["distances"][0] if matches["distances"] else []
        sources = [
            {
                "source": Path(str(metadata.get("source", "document"))).name,
                "page": int(metadata.get("page", 0)) + 1,
                "content": content,
                "similarity": round(1 - float(distance), 4),
            }
            for content, metadata, distance in zip(documents, metadatas, distances)
            if 1 - float(distance) >= 0.0
        ]

        if not sources:
            return {
                "answer": "I couldn't find relevant content in your indexed documents to answer that.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[Source {index}: {source['source']}, page {source['page']}]\n"
            f"{source['content']}"
            for index, source in enumerate(sources, start=1)
        )
        api_key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API is not configured")

        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=0.1,
            max_tokens=1024,
        )
        prompt = f"""You are a helpful assistant answering questions from the user's PDF documents.
Answer accurately and concisely using only the context below. If it does not contain
enough information, say what is missing. Cite factual claims with source numbers like [1].

Context:
{context}

Question: {question}

Answer:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content
        if not isinstance(answer, str):
            answer = "".join(
                part.get("text", "") for part in answer if isinstance(part, dict)
            )
        return {"answer": answer, "sources": sources}