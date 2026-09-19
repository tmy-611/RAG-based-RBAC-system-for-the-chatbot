"""STEP 2 - Data ingestion for the FinSolve RBAC chatbot.

- Walks ``resources/data/<department>/`` (department = folder name).
- ``.md`` files  -> parsed with Docling, then chunked.
- ``.csv`` files -> loaded with LangChain CSVLoader (one Document per row).
- Every chunk carries ``metadata["department"]`` so STEP 3 can enforce
  RBAC with a Chroma ``where`` filter: ``{"department": user_department}``.
- Embeds with HuggingFace ``all-MiniLM-L6-v2`` and persists to ChromaDB
  at ``resources/vectorstore`` (already in .gitignore).

Run from the project root::

    python -m app.utils.ingest            # full re-ingest (idempotent)
    python -m app.utils.ingest --reset    # wipe vectorstore first
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter)
from dotenv import load_dotenv

load_dotenv()


# --- Shared constants (STEP 3 imports these for retrieval) ---
BASE_DIR = Path(__file__).resolve().parents[2]  # <root>/app/utils/ingest.py -> <root>
DATA_DIR = BASE_DIR / "resources" / "data"
PERSIST_DIR = BASE_DIR / "resources" / "vectorstore"
COLLECTION_NAME = "finsolve_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _parent_key(path: Path, metadata: dict) -> str:
    """Section key used to group chunks for small-to-big context expansion."""
    section = metadata.get("h2") or metadata.get("h1") or "root"
    return f"{path.name}::{section}"


def load_markdown(path: Path, department: str, converter: DocumentConverter) -> list[Document]:
    """Parse .md with Docling, split by heading, store parent text per chunk.

    Small chunks are embedded for precise retrieval, while ``parent_text``
    holds the whole parent section so retrieval can expand back to it.
    """
    text = converter.convert(str(path)).document.export_to_markdown()

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON
    )
    overflow_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    sections = header_splitter.split_text(text)

    # Group raw section bodies (and titles) so each chunk can carry its parent.
    parent_titles: dict[str, str] = {}
    parent_bodies: dict[str, list[str]] = {}
    for section in sections:
        key = _parent_key(path, section.metadata)
        parent_titles.setdefault(
            key,
            section.metadata.get("h2") or section.metadata.get("h1") or path.stem,
        )
        parent_bodies.setdefault(key, []).append(section.page_content)

    docs: list[Document] = []
    for section in sections:
        parent_id = _parent_key(path, section.metadata)
        heading_path = " > ".join(str(v) for v in section.metadata.values())
        body = section.page_content
        if heading_path:
            body = f"{heading_path}\n\n{body}"
        section.page_content = body
        section.metadata["department"] = department
        section.metadata["source"] = path.name
        section.metadata["parent_id"] = parent_id
        section.metadata["parent_text"] = (
            f"{parent_titles[parent_id]}\n\n" + "\n\n".join(parent_bodies[parent_id])
        )
        # Header sections can still exceed the embedding window: cap them.
        if len(body) > CHUNK_SIZE:
            docs.extend(overflow_splitter.split_documents([section]))
        else:
            docs.append(section)
    return docs


def load_csv_rows(path: Path, department: str) -> list[Document]:
    """Load a .csv file with CSVLoader; each row is its own parent section."""
    docs = CSVLoader(file_path=str(path), encoding="utf-8").load()
    for doc in docs:
        doc.metadata["department"] = department
        doc.metadata["source"] = path.name
        doc.metadata["parent_id"] = f"{path.name}::row::{doc.metadata.get('row', 0)}"
        doc.metadata["parent_text"] = doc.page_content
    return docs


def build_documents() -> tuple[list[Document], dict[str, int]]:
    """Walk DATA_DIR, return (chunks, per-department file counts)."""
    converter = DocumentConverter()
    chunks: list[Document] = []
    file_counts: dict[str, int] = {}

    for dept_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        department = dept_dir.name
        file_counts[department] = 0
        for path in sorted(dept_dir.iterdir()):
            if path.suffix.lower() == ".md":
                chunks.extend(load_markdown(path, department, converter))
                file_counts[department] += 1
            elif path.suffix.lower() == ".csv":
                chunks.extend(load_csv_rows(path, department))
                file_counts[department] += 1

    return chunks, file_counts


def ingest(reset: bool = False) -> int:
    """Embed all chunks and persist them to ChromaDB. Returns chunk count."""
    if reset and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print(f"Wiped {PERSIST_DIR}")

    chunks, file_counts = build_documents()
    if not chunks:
        raise RuntimeError(f"No documents found under {DATA_DIR}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Idempotent re-runs: drop the old collection before re-adding.
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    if db._collection.count() > 0:
        db.delete_collection()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )

    for dept, n_files in file_counts.items():
        n_chunks = sum(1 for c in chunks if c.metadata.get("department") == dept)
        print(f"  {dept:<12} {n_files} file(s) -> {n_chunks} chunk(s)")
    print(f"Persisted {db._collection.count()} chunks to {PERSIST_DIR}")
    return db._collection.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FinSolve docs into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="wipe vectorstore first")
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
