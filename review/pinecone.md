
## Default cluster setup 
- aws us-east-1
- Everything else is abstracted away

## Limitations
- HTTP response body: {"code":3,"message":"Vector ID must be ASCII, but got 'A Braña_0'","details":[]}
- Pinecone metadata size limit
- No real hybrid search. Use metadata filtering instead

## Cute extra features
- Assistant 
- hosted embedding model (cost?) both for ingest and for inference