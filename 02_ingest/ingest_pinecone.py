"""
Bulk ingest pre-processed Wikipedia chunks into Pinecone.

Usage:
    python ingest_pinecone.py

Environment variables:
    PINECONE_API_KEY - Pinecone API key
    PINECONE_INDEX_NAME - Target index name
"""

import os
import json
import hashlib
from pathlib import Path
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_DIM = 384
BATCH_SIZE = 100  # Pinecone recommends 100 vectors per upsert
MAX_VECTORS = 10_000  # Limit total vectors to ingest
DATA_DIR = Path(__file__).parent.parent / "data" / "converted"
PROGRESS_FILE = Path(__file__).parent / "ingest_pinecone_done.txt"


def load_completed_batches() -> set[str]:
    """Load set of already completed batch filenames."""
    if not PROGRESS_FILE.exists():
        return set()
    return set(PROGRESS_FILE.read_text().strip().split("\n"))


def mark_batch_complete(filename: str):
    """Append completed batch filename to progress file."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{filename}\n")


def create_index_if_not_exists(pc: Pinecone):
    """Create index if it doesn't exist."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists")
        return

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
    print(f"Created index '{PINECONE_INDEX_NAME}'")


def sanitize_id(doc_id: str) -> str:
    """Convert ID to ASCII-safe format using hash if needed."""
    # Try ASCII encoding - if it fails, use hash
    try:
        doc_id.encode('ascii')
        return doc_id
    except UnicodeEncodeError:
        # Use hash for non-ASCII IDs
        hash_hex = hashlib.md5(doc_id.encode('utf-8')).hexdigest()
        return f"doc_{hash_hex}"


def load_vectors_from_batch(filepath: Path) -> list[dict]:
    """Load vectors from a single JSONL batch file."""
    vectors = []

    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line)

            # Skip index action lines (they have "index" key)
            if "index" in doc:
                continue

            vectors.append({
                "id": sanitize_id(doc["id"]),
                "values": doc["embedding"],
                "metadata": {
                    "title": doc["title"],
                    "text": doc["text"],
                    "chunk_index": doc["chunk_index"],
                    "text_length": doc["text_length"],
                    "original_id": doc["id"]  # Keep original for reference
                }
            })

    return vectors


def ingest_batch(index, filepath: Path) -> tuple[int, int]:
    """Ingest a single batch file. Returns (success_count, error_count)."""
    vectors = load_vectors_from_batch(filepath)
    success_count = 0
    error_count = 0

    # Upsert in smaller batches
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        try:
            index.upsert(vectors=batch)
            success_count += len(batch)
        except Exception as e:
            error_count += len(batch)
            print(f"Error: {e}")

    return success_count, error_count


def main():
    if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
        raise ValueError("Missing required environment variables")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if needed
    create_index_if_not_exists(pc)

    # Get index
    index = pc.Index(PINECONE_INDEX_NAME)

    # Check index stats
    stats = index.describe_index_stats()
    current_count = stats.total_vector_count
    print(f"Index stats: {current_count:,} vectors")

    # Calculate how many more vectors we can add
    remaining_capacity = MAX_VECTORS - current_count
    if remaining_capacity <= 0:
        print(f"Index already has {current_count:,} vectors (limit: {MAX_VECTORS:,}). Nothing to ingest!")
        return

    print(f"Will add up to {remaining_capacity:,} more vectors to reach limit of {MAX_VECTORS:,}")

    # Get batch files and filter out completed ones
    all_batch_files = sorted(DATA_DIR.glob("elasticsearch_batch_*.jsonl"))
    completed = load_completed_batches()
    batch_files = [f for f in all_batch_files if f.name not in completed]

    print(f"Found {len(all_batch_files)} total batch files")
    print(f"Skipping {len(completed)} already completed batches")
    print(f"Processing {len(batch_files)} remaining batches")

    if not batch_files:
        print("Nothing to ingest!")
        return

    # Ingest each batch and track progress
    total_success = 0
    total_errors = 0

    for i, filepath in enumerate(batch_files):
        # Check if we've reached the limit
        if current_count + total_success >= MAX_VECTORS:
            print(f"\nReached limit of {MAX_VECTORS:,} vectors. Stopping.")
            break

        print(f"[{i+1}/{len(batch_files)}] Processing {filepath.name}")

        success, errors = ingest_batch(index, filepath)
        total_success += success
        total_errors += errors

        # Mark batch as complete
        mark_batch_complete(filepath.name)
        print(f"  -> {success:,} docs indexed, {errors} errors")

    print(f"\nIngestion complete: {total_success:,} vectors indexed")
    if total_errors:
        print(f"Total errors: {total_errors:,}")

    # Final stats
    stats = index.describe_index_stats()
    print(f"Final index stats: {stats.total_vector_count:,} vectors")


if __name__ == "__main__":
    main()
