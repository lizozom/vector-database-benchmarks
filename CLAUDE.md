# Vector Database Benchmark Project

A benchmark suite for comparing cloud vector database providers using a standardized dataset and naive configuration to establish baseline performance metrics.

## Project Goal

Evaluate vector databases on a level playing field by using **default configurations** across all providers. This intentionally naive approach:
1. Establishes baseline performance for each provider
2. Identifies which providers offer the best out-of-the-box experience
3. Sets the stage for future optimization experiments with better chunking/embedding strategies

## Benchmark Methodology

### Dataset
- **Source**: Wikipedia dataset (~250k articles, A-prefixed subset) from https://www.kaggle.com/datasets/jjinho/wikipedia-20230701
- **Chunking**: Fixed-size chunks (1000 chars with 100 char overlap)
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, max 256 tokens)
  - 1000 chars ≈ 200-280 tokens, reasonably within model limit
- **Pre-computed embeddings**: Stored in `data/converted/` as JSONL batches

### Evaluation Criteria
| Criterion | Description |
|-----------|-------------|
| Developer Experience | SDK quality, docs, ease of setup |
| Retrieval Quality (MAP) | Mean Average Precision (expected to be similar across providers) |
| Performance Metrics | Ingestion speed, query latency, throughput (TBD) |
| Cost | Pricing at scale for storage and queries |

## Project Structure

```
database-reviews/
├── 01_pre-process/     # Wikipedia data download and chunking scripts
├── 02_ingest/          # Provider-specific ingestion scripts
├── 03_search/          # Search/query benchmarking scripts
├── data/               # Pre-computed embedding batches
└── .env                # API keys and configuration
```

## Vector DB Providers to Evaluate

- Pinecone
- pgvector on Supabase
- Elasticsearch Cloud
- Milvus / Zilliz
- Weaviate

## Development Guidelines

### Adding a New Provider
1. Create ingestion script in `02_ingest/<provider>_ingest.py`
2. Create search benchmark in `03_search/<provider>_search.py`
3. Use the pre-computed embeddings from `embeddings/` directory
4. Document any provider-specific setup in the script

### Environment Variables
Store all API keys and endpoints in `.env`. Never commit secrets.

