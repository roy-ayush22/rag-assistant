from __future__ import annotations

import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient, models

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
QDRANT_INFERENCE_MODEL = os.getenv(
    "QDRANT_INFERENCE_MODEL",
    "sentence-transformers/all-minilm-l6-v2",
)
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "electric_motors")
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
NO_ANSWER = "I couldn't find that in the electric motor reference material."
SYSTEM_PROMPT = r"""\
You are the Electric Motor Assistant for a Retrieval-Augmented Generation
system. Answer the user's current question using the supplied reference
passages. Recent conversation may clarify a follow-up question, but the
passages are the source of truth.

Rules:
- Be accurate, direct, and approachable for an engineering student.
- Answer only what the passages support. Do not invent specifications,
  calculations, citations, or document details.
- If the passages do not answer the question, reply exactly:
  "I couldn't find that in the electric motor reference material."
- Cite factual claims with the relevant passage number in square brackets,
  for example [1] or [1][3]. Do not cite unsupported claims.
- Explain a technical term briefly when it improves clarity. Use short
  headings or bullets only when they make the answer easier to follow.
- Treat the latest user question as the requested task. Use conversation
  history only to resolve references such as "it", "that motor", or
  "the previous equation".

Math formatting contract:
- Write every equation or symbolic expression in LaTeX.
- Use \( ... \) for inline math and \[ ... \] on their own line for display
  math. Never use $ or $$ delimiters.
- Keep normal prose, units, source citations, and monetary amounts outside
  math delimiters unless they are part of an equation.
- Use LaTeX commands such as \frac, \sqrt, subscripts, superscripts, and
  \mathrm where appropriate. Do not put math in code fences.
- After a display equation, explain the symbols or relationship in plain
  language when useful.
"""


def _format_bare_math(text: str) -> str:
    """Normalise math delimiters so remark-math / KaTeX can parse them reliably.

    remark-math understands both $ / $$ and \\( / \\[ delimiters.
    We convert everything to the \\( ... \\) / \\[ ... \\] form which is
    unambiguous (no collision with currency signs or stray dollar characters).
    """
    # 1. $$...$$ display math  →  \\[ ... \\]
    text = re.sub(
        r"\$\$([\s\S]+?)\$\$",
        lambda m: "\\[\n" + m.group(1).strip() + "\n\\]",
        text,
    )
    # 2. $...$ inline math  →  \\( ... \\)
    #    Guard: don't match a lone $ (e.g. currency) – require at least one
    #    LaTeX-looking character inside (\, ^, _, digits mixed with letters).
    text = re.sub(
        r"\$([^$\n]{1,300}?)\$",
        lambda m: "\\(" + m.group(1).strip() + "\\)"
        if re.search(r"[\\^_{}]|\\[A-Za-z]+|\d", m.group(1))
        else m.group(0),
        text,
    )

    # --- helper: split text into math-spans vs plain-text segments ---
    _MATH_SPAN = re.compile(r"(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\))")

    def _apply_outside_math(text: str, fn) -> str:
        """Apply `fn` only to segments that are NOT inside \\(…\\) or \\[…\\]."""
        parts = _MATH_SPAN.split(text)
        for i in range(0, len(parts), 2):  # even indices = plain text
            parts[i] = fn(parts[i])
        return "".join(parts)

    # 3. Bare [formula] brackets  →  \\( formula \\)
    _BARE_BRACKET = re.compile(
        r"(?<!\\)\[\s*([^\[\]\n]*?(?:\\[A-Za-z]+|[=_^])[^\[\]\n]*?)\s*(?<!\\)\]"
    )
    text = _apply_outside_math(text, lambda s: _BARE_BRACKET.sub(
        lambda m: "\\(" + m.group(1).strip().rstrip(" ;,.") + "\\)", s
    ))

    # 4. Catch parenthesised math variables: (I_a), (R_a/k^2), (V_{dc})
    #    The outer parentheses are REPLACED by \( \).
    #    Must contain a subscript/superscript to avoid false positives on (A) or (k).
    _PAREN_MATH = re.compile(
        r"\(\s*("
        r"[A-Za-z][A-Za-z0-9]*[_^]\{?[A-Za-z0-9]+\}?"  # e.g. I_a, V_{dc}, k^2
        r"(?:\s*/\s*[A-Za-z0-9_^{}]+)*"                  # optional /fraction
        r")\s*\)"
    )
    text = _apply_outside_math(text, lambda s: _PAREN_MATH.sub(
        lambda m: "\\(" + m.group(1).strip() + "\\)", s
    ))

    # 5. Catch remaining plain-text math tokens NOT inside parentheses.
    #    Patterns: "R_a", "I_a", "k^2", "V_{dc}", "R_a/k^2"
    #    Only match isolated tokens that look mathematical (contain _ or ^).
    _PLAIN_MATH_TOKEN = re.compile(
        r"(?<![\\$a-zA-Z(])"                       # not inside a word, parens, or already-delimited
        r"("
        r"[A-Za-z][A-Za-z0-9]*"                    # base letter(s)
        r"(?:[_^]\{?[A-Za-z0-9]+\}?)+"             # at least one sub/superscript
        r"(?:\s*/\s*[A-Za-z][A-Za-z0-9]*"          # optional "/fraction" part
        r"(?:[_^]\{?[A-Za-z0-9]+\}?)*)?"           #   with its own sub/superscripts
        r")"
        r"(?![a-zA-Z\\)])"                         # not followed by more word chars or closing paren
    )
    text = _apply_outside_math(text, lambda s: _PLAIN_MATH_TOKEN.sub(
        lambda m: "\\(" + m.group(1) + "\\)", s
    ))

    return text


def _normalise_math_delimiters(text: str) -> str:
    """Convert legacy dollar-delimited LaTeX without touching ordinary prose."""
    text = re.sub(
        r"\$\$([\s\S]+?)\$\$",
        lambda match: "\\[\n" + match.group(1).strip() + "\n\\]",
        text,
    )

    def replace_inline_math(match: re.Match[str]) -> str:
        value = match.group(1).strip()
        is_latex = "\\" in value or any(symbol in value for symbol in "_^={}")
        return f"\\({value}\\)" if value and is_latex else match.group(0)

    return re.sub(r"(?<!\$)\$([^$\n]{1,300}?)\$(?!\$)", replace_inline_math, text)


class RagService:
    """Queries the shared Qdrant collection and answers user questions."""

    def __init__(self) -> None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be configured")

        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._client = None
        self._client_lock = Lock()
        self.collection_name = QDRANT_COLLECTION

    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = QdrantClient(
                        url=self._qdrant_url,
                        api_key=self._qdrant_api_key,
                        cloud_inference=True,
                        check_compatibility=False,
                    )
        return self._client

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def ask(
        self,
        question: str,
        top_k: int = 5,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        collection_count = self.count()
        if collection_count == 0:
            return {"answer": NO_ANSWER, "sources": []}

        history = (history or [])[-8:]
        api_key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
        llm = None
        retrieval_question = question
        if history and api_key:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=DEFAULT_GROQ_MODEL,
                temperature=0.1,
                max_tokens=2048,
            )
            conversation = "\n".join(
                f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
            )
            try:
                rewritten = llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Rewrite the latest user question as a standalone search query, "
                                "using the conversation only to resolve references such as 'it' "
                                "or 'that motor'. Preserve the user's intent. Return only the "
                                "query; do not answer it. If it is already standalone, return it unchanged."
                            )
                        ),
                        HumanMessage(
                            content=f"Conversation:\n{conversation}\n\nLatest question: {question}"
                        ),
                    ]
                )
                rewritten_content = rewritten.content
                if isinstance(rewritten_content, str) and rewritten_content.strip():
                    retrieval_question = rewritten_content.strip().strip('"')
            except Exception:
                # Keep the original query if clarification fails; normal RAG can still proceed.
                pass

        matches = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=retrieval_question,
                model=QDRANT_INFERENCE_MODEL,
            ),
            limit=min(top_k, collection_count),
            with_payload=True,
        )
        sources = []
        for point in matches.points:
            payload = point.payload or {}
            similarity = float(point.score or 0.0)
            if similarity < MIN_SIMILARITY:
                continue
            sources.append(
                {
                    "source": Path(str(payload.get("source", "document"))).name,
                    "page": int(payload.get("page", 0)) + 1,
                    "content": str(payload.get("text", "")),
                    "similarity": round(similarity, 4),
                }
            )

        if not sources:
            return {"answer": NO_ANSWER, "sources": []}

        if not api_key:
            raise RuntimeError("GROQ_API is not configured")
        if llm is None:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=DEFAULT_GROQ_MODEL,
                temperature=0.1,
                max_tokens=2048,
            )

        labeled_context = "\n\n".join(
            f"[{index}] {source['content']}"
            for index, source in enumerate(sources, start=1)
        )
        conversation_context = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
        )
        history_section = (
            f"Recent conversation (use only to understand references in the latest question):\n{conversation_context}\n\n"
            if conversation_context
            else ""
        )
        prompt = f"""Use the following retrieved passages to answer the question.
If none of the passages contains the answer, reply exactly: {NO_ANSWER}
Include the supporting passage number, such as [1].
Write equations in LaTeX. Use \\( ... \\) for inline math and \\[ ... \\] on separate lines for display math. Do NOT use $ or $$ as math delimiters.

{history_section}Passages:
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
        answer = _normalise_math_delimiters(answer)
        return {"answer": answer, "sources": sources}
