"""
Bulk ingest pre-processed Wikipedia chunks into Elasticsearch Cloud.

Usage:
    python elasticsearch.py

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


def create_index(client: Elasticsearch):
    """Create index with dense_vector mapping if it doesn't exist."""
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists")
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


def generate_actions(batch_files: list[Path]):
    """Generate bulk actions from JSONL batch files."""
    for filepath in batch_files:
        print(f"Processing {filepath.name}")

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

    # Create index
    create_index(client)

    # Get batch files
    batch_files = sorted(DATA_DIR.glob("elasticsearch_batch_*.jsonl"))
    print(f"Found {len(batch_files)} batch files")

    # Bulk ingest with progress tracking
    success_count = 0
    error_count = 0

    for ok, result in parallel_bulk(
        client,
        generate_actions(batch_files),
        chunk_size=1000,
        thread_count=4,
        raise_on_error=False
    ):
        if ok:
            success_count += 1
        else:
            error_count += 1
            print(f"Error: {result}")

        if success_count % 10000 == 0:
            print(f"Progress: {success_count:,} documents indexed")

    print(f"\nIngestion complete: {success_count:,} documents indexed")
    if error_count:
        print(f"Errors: {error_count:,}")

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
