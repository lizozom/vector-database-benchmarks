"""
Bulk ingest pre-processed Wikipedia chunks into Elasticsearch Cloud.

Usage:
    python ingest_elasticsearch.py

Environment variables:
    ELASTICSEARCH_ENDPOINT - Elasticsearch Cloud endpoint URL
    ELASTICSEARCH_API_KEY - API key for authentication
    INDEX_NAME - Target index name
"""

import os
import json
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from dotenv import load_dotenv

load_dotenv()

ELASTICSEARCH_ENDPOINT = os.getenv("ELASTICSEARCH_ENDPOINT")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME").lower()

EMBEDDING_DIM = 384
DATA_DIR = Path(__file__).parent.parent / "data" / "converted"
PROGRESS_FILE = Path(__file__).parent / "ingest_elasticsearch_done.txt"


def load_completed_batches() -> set[str]:
    """Load set of already completed batch filenames."""
    if not PROGRESS_FILE.exists():
        return set()
    return set(PROGRESS_FILE.read_text().strip().split("\n"))


def mark_batch_complete(filename: str):
    """Append completed batch filename to progress file."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{filename}\n")


def create_or_prepare_index(client: Elasticsearch):
    """Create index or prepare existing index for bulk ingestion."""
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists")
        # Disable refresh and replicas for faster ingestion
        print("Disabling refresh and replicas for ingestion...")
        client.indices.put_settings(index=INDEX_NAME, body={
            "index": {
                "refresh_interval": "-1",
                "number_of_replicas": 0
            }
        })
        return

    mappings = {
        "settings": {
            "number_of_replicas": 0,
            "refresh_interval": "-1"
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "chunk_index": {"type": "integer"},
                "text_length": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
    }

    client.indices.create(index=INDEX_NAME, body=mappings)
    print(f"Created index '{INDEX_NAME}' (refresh disabled, 0 replicas)")


def generate_actions_for_batch(filepath: Path):
    """Generate bulk actions from a single JSONL batch file."""
    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line)

            # Skip index action lines (they have "index" key)
            if "index" in doc:
                continue

            yield {
                "_index": INDEX_NAME,
                "_id": doc["id"],
                "_source": doc
            }


def ingest_batch(client: Elasticsearch, filepath: Path) -> tuple[int, int]:
    """Ingest a single batch file. Returns (success_count, error_count)."""
    success_count = 0
    error_count = 0

    for ok, result in parallel_bulk(
        client,
        generate_actions_for_batch(filepath),
        chunk_size=1000,
        thread_count=1,
        raise_on_error=False
    ):
        if ok:
            success_count += 1
        else:
            error_count += 1
            print(f"Error: {result}")

    return success_count, error_count


def main():
    if not all([ELASTICSEARCH_ENDPOINT, ELASTICSEARCH_API_KEY, INDEX_NAME]):
        raise ValueError("Missing required environment variables")

    client = Elasticsearch(
        ELASTICSEARCH_ENDPOINT,
        api_key=ELASTICSEARCH_API_KEY
    )

    # Verify connection
    info = client.info()
    print(f"Connected to Elasticsearch: {info['version']['number']}")

    # Create or prepare index
    create_or_prepare_index(client)

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
        print(f"[{i+1}/{len(batch_files)}] Processing {filepath.name}")

        success, errors = ingest_batch(client, filepath)
        total_success += success
        total_errors += errors

        # Mark batch as complete
        mark_batch_complete(filepath.name)
        print(f"  -> {success:,} docs indexed, {errors} errors")

    print(f"\nIngestion complete: {total_success:,} documents indexed")
    if total_errors:
        print(f"Total errors: {total_errors:,}")

    # Restore normal settings
    print("Restoring index settings...")
    client.indices.put_settings(index=INDEX_NAME, body={
        "index": {
            "refresh_interval": "1s",
            "number_of_replicas": 1
        }
    })
    client.indices.refresh(index=INDEX_NAME)
    print("Index settings restored and refreshed")


if __name__ == "__main__":
    main()
