# Vector Database Benchmark

Benchmarking cloud vector database providers using Wikipedia dataset with naive configuration.

## Setup

```bash
poetry install
cp .env.example .env  # Edit with your credentials
```

## Environment Variables

```
ELASTICSEARCH_ENDPOINT=https://...
ELASTICSEARCH_API_KEY=...
INDEX_NAME=your-index-name
```

## Usage

### Ingest to Elasticsearch

```bash
poetry run python 02_ingest/ingest_elasticsearch.py
```

