# FAISS Prototype Demo

This project demonstrates basic vector similarity search using Facebook AI Similarity Search (FAISS).

## Files
- `retrieval_demo.py`: Python script with FAISS example.

## Requirements
- faiss-cpu
- numpy

## Run
```bash  
python retrieval_demo.py

How Vector Search Works ?

Vector search uses machine learning to convert data into numerical representations called vectors, allowing for semantic similarity search. This approach moves beyond keyword matching, enabling users to find relevant results based on the meaning and context of their search query

Word embeddings represent words as dense vectors in a continuous vector space.

These vectors are learned based on context — words used in similar contexts have similar embeddings.

Embeddings capture semantic meaning — e.g., "king" - "man" + "woman" ≈ "queen".

Cosine similarity measures how similar two vectors are by comparing the angle between them.

In sentence embeddings, full sentences are represented in a similar way using models like Sentence-BERT.

Vector search involves:

Converting a query into a vector.

Comparing it to all stored vectors (e.g., in FAISS index).

Returning the top-k closest vectors, which correspond to the most semantically similar sentences.

This allows semantic search — finding relevant results even when exact words differ.