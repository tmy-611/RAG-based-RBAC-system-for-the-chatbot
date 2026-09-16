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
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Shared constants (STEP 3 imports these for retrieval) ---
BASE_DIR = Path(__file__).resolve().parents[2]  # <root>/app/utils/ingest.py -> <root>
DATA_DIR = BASE_DIR / "resources" / "data"
PERSIST_DIR = BASE_DIR / "resources" / "vectorstore"
COLLECTION_NAME = "finsolve_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_markdown(path: Path, department: str, converter: DocumentConverter) -> str:
    """Parse a .md file with Docling, return clean markdown text."""
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def load_csv_rows(path: Path, department: str) -> list[Document]:
    """Load a .csv file with CSVLoader, tagging every row with its department."""
    docs = CSVLoader(file_path=str(path), encoding="utf-8").load()
    for doc in docs:
        doc.metadata["department"] = department
        doc.metadata["source"] = path.name
    return docs


def build_documents() -> tuple[list[Document], dict[str, int]]:
    """Walk DATA_DIR, return (chunks, per-department file counts)."""
    converter = DocumentConverter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[Document] = []
    file_counts: dict[str, int] = {}

    for dept_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        department = dept_dir.name
        file_counts[department] = 0
        for path in sorted(dept_dir.iterdir()):
            if path.suffix.lower() == ".md":
                text = load_markdown(path, department, converter)
                if not text.strip():
                    continue
                doc = Document(
                    page_content=text,
                    metadata={"department": department, "source": path.name},
                )
                chunks.extend(splitter.split_documents([doc]))
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
