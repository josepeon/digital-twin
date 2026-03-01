#!/usr/bin/env python3
"""
RAG index builder — embed all parsed messages into ChromaDB.

Reads a versioned dataset JSONL and embeds every message into a ChromaDB
persistent collection with metadata for filtered retrieval.

Usage:
    python scripts/build_rag_index.py [--dataset data/personal/processed/v0.1.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    EMBEDDING_MODEL,
    PROCESSED_DIR,
    RAG_COLLECTION_NAME,
    RAG_INDEX_DIR,
)


def build_rag_index(
    dataset_path: Path,
    index_dir: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    batch_size: int = 256,
    rebuild: bool = False,
) -> int:
    """Build or update the ChromaDB RAG index from a dataset JSONL.

    Args:
        dataset_path: Path to the versioned dataset JSONL.
        index_dir: ChromaDB persistence directory.
        collection_name: Name for the collection.
        embedding_model: Sentence-transformer model name.
        batch_size: Number of documents per batch insert.
        rebuild: If True, delete existing collection and rebuild from scratch.

    Returns:
        Number of documents indexed.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    idx_dir = index_dir or RAG_INDEX_DIR
    col_name = collection_name or RAG_COLLECTION_NAME
    emb_model = embedding_model or EMBEDDING_MODEL

    print(f"🗄️  Index directory: {idx_dir}")
    print(f"🔤 Embedding model: {emb_model}")
    print(f"📚 Collection: {col_name}")

    # Load dataset
    print(f"\n📂 Loading dataset: {dataset_path}")
    entries: list[dict] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("❌ Dataset is empty.")
        return 0

    print(f"   {len(entries)} entries loaded")

    # Initialize ChromaDB
    idx_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(idx_dir))

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=emb_model,
    )

    if rebuild:
        try:
            client.delete_collection(col_name)
            print(f"   🗑️  Deleted existing collection '{col_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=col_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    print(f"   Existing documents in collection: {existing_count}")

    # Prepare documents for insertion
    # We store Jose's text as the document, with context + metadata.
    # For RAG retrieval, we want the model to find messages similar to the user's query,
    # then use Jose's actual response as the few-shot example.
    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for entry in entries:
        doc_id = entry.get("id", "")
        text = entry.get("text", "")
        if not text or not doc_id:
            continue

        # Skip if already indexed (unless rebuilding)
        if not rebuild and doc_id in ids:
            continue

        docs.append(text)
        metadatas.append({
            "source": entry.get("source", "unknown"),
            "register": entry.get("register", "neutral"),
            "language": entry.get("language", "en"),
            "context": entry.get("context", "")[:500],  # Truncate long contexts
            "quality_score": entry.get("quality_score", 0.0),
            "timestamp": entry.get("timestamp", ""),
        })
        ids.append(doc_id)

    if not docs:
        print("⚠ No new documents to index.")
        return existing_count

    # Batch insert
    total = len(docs)
    indexed = 0
    print(f"\n⏳ Indexing {total} documents (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        try:
            collection.upsert(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )
            indexed += len(batch_docs)
            pct = (indexed / total) * 100
            print(f"   [{indexed}/{total}] {pct:.0f}%")
        except Exception as e:
            print(f"   ⚠ Batch {i}-{i + batch_size} failed: {e}")

    final_count = collection.count()
    print(f"\n✅ RAG index built: {final_count} total documents in '{col_name}'")

    # Quick sanity check — query with a test string
    test_query = "What are you working on?"
    results = collection.query(query_texts=[test_query], n_results=3)
    print(f"\n🔍 Sanity check: '{test_query}'")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        preview = doc[:100] + "..." if len(doc) > 100 else doc
        print(f"   [{i + 1}] ({meta['source']}/{meta['register']}) {preview}")

    return final_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB RAG index from dataset")
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=PROCESSED_DIR / "v0.1.jsonl",
        help="Path to dataset JSONL",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=f"ChromaDB directory (default: {RAG_INDEX_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=f"Collection name (default: {RAG_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=f"Sentence-transformer model (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for indexing",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the collection from scratch",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        print("   Run build_dataset.py first to create a versioned dataset.")
        sys.exit(1)

    build_rag_index(
        dataset_path=args.dataset,
        index_dir=args.index_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
