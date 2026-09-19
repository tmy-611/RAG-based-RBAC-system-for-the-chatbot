"""RAG retrieval with RBAC metadata filtering and small-to-big expansion."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.services.guardrails import OUT_OF_SCOPE_REPLY, is_in_scope, mask_pii
from app.utils.ingest import COLLECTION_NAME, EMBED_MODEL, PERSIST_DIR

load_dotenv()

TOP_K = 4
ANSWER_MODEL = "openai/gpt-oss-120b"
NO_CONTEXT_REPLY = (
    "I couldn't find relevant information in the documents you have access to"
)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Open the persisted Chroma collection once, then reuse it."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


def build_filter(role: str) -> dict | None:
    """Map a user role to a Chroma where-filter. C_Level sees everything."""
    if role.lower() == "c_level":
        return None
    return {"department": role}


def retrieve(query: str, role: str, k: int = TOP_K) -> list[Document]:
    """Small-to-big retrieval: match small chunks, return their parent sections.

    Vector search finds precise small chunks, then each hit is expanded to
    its full parent section so the LLM gets self-sufficient context. Repeated
    parents are de-duplicated, so overlapping hits do not repeat text.
    """
    hits = get_vectorstore().similarity_search(query, k=k, filter=build_filter(role))

    seen: set[str] = set()
    contexts: list[Document] = []
    for hit in hits:
        parent_id = hit.metadata.get("parent_id") or str(id(hit))
        if parent_id in seen:
            continue
        seen.add(parent_id)
        contexts.append(
            Document(
                page_content=hit.metadata.get("parent_text") or hit.page_content,
                metadata=hit.metadata,
            )
        )
    return contexts


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatGroq:
    """Main answering LLM: strong reasoning, deterministic output."""
    return ChatGroq(model=ANSWER_MODEL, temperature=0)


def build_prompt(question: str, chunks: list[Document]) -> str:
    """Pack retrieved context as the LLM's only source of truth."""
    context = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)
    return (
        "You are FinSolve's internal assistant. Answer the question using ONLY "
        "the context below. If the context lacks the answer, say so plainly. "
        "When refusing or stating lack of information, do not repeat personal "
        "names, emails, or identifiers from the question. Refer to 'the person asked about' instead. "
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def answer_question(message: str, role: str) -> str:
    """Full pipeline: route -> retrieve -> generate -> mask."""
    question = message.strip()
    if not question:
        return "Please enter a question."
    if not is_in_scope(question):
        return OUT_OF_SCOPE_REPLY
    chunks = retrieve(question, role)
    if not chunks:
        return NO_CONTEXT_REPLY
    raw_answer = get_answer_llm().invoke(build_prompt(question, chunks)).content
    return mask_pii(raw_answer)
