from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.rag import RagService


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.rag = RagService()
    yield


app = FastAPI(title="RAG Document Assistant", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question cannot be blank")
        return question


class Source(BaseModel):
    source: str
    page: int
    content: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "RAG API is running", "docs": "/docs"}


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    service: RagService = request.app.state.rag
    return {"status": "ok", "indexed_chunks": service.collection.count()}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
    service: RagService = request.app.state.rag
    if service.collection.count() == 0:
        return {
            "answer": "No PDF content is indexed yet. Add a PDF to the data folder and restart the API.",
            "sources": [],
        }

    try:
        return service.ask(payload.question.strip(), payload.top_k)
    except RuntimeError as error:
        if "GROQ_API" in str(error):
            raise HTTPException(
                status_code=503,
                detail="Set GROQ_API (or GROQ_API_KEY) in the project .env file to enable answers.",
            ) from error
        raise HTTPException(status_code=502, detail="The RAG service could not answer this question.") from error
    except Exception as error:
        raise HTTPException(
            status_code=502, detail="The RAG service could not answer this question."
        ) from error
