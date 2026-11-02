from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
load_dotenv(ROOT_DIR / ".env")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
NO_ANSWER = "I couldn't find that in the electric motor reference material."


def _format_bare_math(text: str) -> str:
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda match: "$$\n" + match.group(1).strip() + "\n$$",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\\((.*?)\\\)",
        lambda match: "$" + match.group(1).strip() + "$",
        text,
        flags=re.DOTALL,
    )

    def wrap_formula(match: re.Match[str]) -> str:
        formula = match.group(1).strip().rstrip(" ;,.")
        return f"${formula}$"

    return re.sub(
        r"(?<!\\)\[\s*([^\[\]\n]*?(?:\\[A-Za-z]+|=)[^\[\]\n]*?)\s*(?<!\\)\]",
        wrap_formula,
        text,
    )


class RagService:
    """Loads the notebook's persisted Chroma index and answers user questions."""

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        pdfs = sorted(DATA_DIR.rglob("*.pdf"))
        fingerprint = hashlib.sha256(EMBEDDING_MODEL.encode("utf-8"))
        for pdf in pdfs:
            fingerprint.update(str(pdf.relative_to(DATA_DIR)).encode("utf-8"))
            fingerprint.update(pdf.read_bytes())
        collection_name = f"pdf_docs_{fingerprint.hexdigest()[:12]}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
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
            n_results=top_k,
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
            if 1 - float(distance) >= MIN_SIMILARITY
        ]

        if not sources:
            return {"answer": NO_ANSWER, "sources": []}

        api_key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API is not configured")

        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=DEFAULT_GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )
        labeled_context = "\n\n".join(
            f"[{index}] {source['content']}"
            for index, source in enumerate(sources, start=1)
        )
        prompt = f"""You answer questions about electric motors from the reference passages below.
Use only facts supported by these passages. Match the meaning of the question even when it is phrased simply or uses different words than the passage.
Answer directly when a passage contains the answer. Do not abstain merely because the question is short.
If none of the passages contains the answer, reply exactly: {NO_ANSWER}
Include the supporting passage number, such as [1].
Write equations in LaTeX using $...$ inline or $$...$$ on separate lines; do not use bare square brackets as math delimiters.

Passages:
{labeled_context}

Question: {question}
Answer:"""
        response = llm.invoke([prompt])
        answer = response.content
        if not isinstance(answer, str):
            answer = "".join(
                part.get("text", "") for part in answer if isinstance(part, dict)
            )
        answer = answer.strip()
        if answer == NO_ANSWER and sources[0]["similarity"] >= 0.50:
            answer = f"{sources[0]['content'].strip()} [1]"
        answer = _format_bare_math(answer)
        return {"answer": answer, "sources": sources}
