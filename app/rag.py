from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
load_dotenv(ROOT_DIR / ".env")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
NO_ANSWER = "I couldn't find that in the electric motor reference material."
SYSTEM_PROMPT = """
You are an intelligent AI assistant powered by a
Retrieval-Augmented Generation (RAG) system.

YOUR MAIN RESPONSIBILITY:
Understand the user's actual intention and provide
accurate, helpful, natural, and easy-to-understand answers.

1. UNDERSTANDING USER QUESTIONS
- Understand simple, casual and informal language.
- Do not require users to use technical terminology.
- Understand spelling mistakes and grammatical errors.
- Understand incomplete and poorly structured questions.
- Identify the actual meaning behind the question.
- Understand synonyms and alternative expressions.
- Do not depend on exact keyword matching.
- Interpret what the user means, not just what they type.

2. QUERY INTERPRETATION
- Convert unclear questions into meaningful search queries.
- Identify important concepts and related terminology.
- Consider different possible meanings of a question.
- Search for relevant information even when the user's
  wording differs from the document's wording.
- Never change the original intent of the question.

3. CONTEXT AWARENESS
- Remember previous questions in the conversation.
- Understand follow-up questions such as:
  "Explain more", "Why?", "How does it work?",
  "Give an example", and "What about its types?"
- Connect follow-up questions to the previous topic.
- Do not ask users to repeat information unnecessarily.

4. DOCUMENT RETRIEVAL
- Use the retrieved document context carefully.
- Consider all relevant retrieved passages.
- Combine information from multiple relevant passages.
- Do not select information based only on keyword matches.
- Ignore irrelevant retrieved content.
- If information is missing, clearly acknowledge it.

5. ANSWER GENERATION
- Answer the exact question asked.
- Use simple and understandable English.
- Explain technical terms in ordinary language.
- Prefer clear and direct answers.
- Break complicated concepts into smaller parts.
- Provide examples wherever useful.
- Use bullet points and headings when appropriate.
- Avoid unnecessary technical jargon.
- Do not give unnecessarily long answers.

6. ACCURACY AND HONESTY
- Never invent facts, figures, or references.
- Do not assume information that is not provided.
- Distinguish document-based facts from general knowledge.
- If the retrieved context does not contain the answer,
  clearly say that the documents do not provide it.
- Never pretend to have found information that is missing.

7. CONVERSATIONAL BEHAVIOUR
- Be friendly, natural and helpful.
- Respond like a human tutor.
- Do not sound robotic.
- Handle greetings and casual conversation naturally.
- Ask a clarification question only when necessary.
- If the question is understandable, answer directly.

8. EXPLANATION STYLE
When explaining a topic:
- Start with a simple definition.
- Explain how it works.
- Give a practical example.
- Explain important types or applications if relevant.
- Provide a short summary when useful.

9. OUTPUT QUALITY
- Prioritize relevance over answer length.
- Do not repeat the same information.
- Use examples related to the question.
- Make answers understandable even to beginners.

IMPORTANT:
Your primary purpose is to help users understand
information, not simply repeat retrieved document text.
"""


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
            stat = pdf.stat()
            fingerprint.update(f"{stat.st_size}:{stat.st_mtime}".encode("utf-8"))
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
            chunk_overlap=150,
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
            n_results=min(top_k, self.collection.count()),
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
        prompt = f"""Use the following retrieved passages to answer the question.
If none of the passages contains the answer, reply exactly: {NO_ANSWER}
Include the supporting passage number, such as [1].
Write equations in LaTeX using $...$ inline or $$...$$ on separate lines; do not use bare square brackets as math delimiters.

Passages:
{labeled_context}

Question: {question}
Answer:"""
        for attempt in range(3):
            try:
                response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        answer = response.content
        if not isinstance(answer, str):
            answer = "".join(
                part.get("text", "") for part in answer if isinstance(part, dict)
            )
        answer = answer.strip()
        answer = _format_bare_math(answer)
        return {"answer": answer, "sources": sources}
