from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example text chunks
sentences = [
    "It is raining today.",
    "The weather is gloomy and wet.",
    "The sun is shining brightly.",
    "Dogs enjoy running in parks.",
    "The cat sits on the mat."
]

# Embed and build FAISS index
embeddings = model.encode(sentences)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Rephrasings of the same question
queries = [
    "What’s the weather like today?",
    "Is it raining or sunny?",
    "Can you tell me about the weather?",
    "Do I need an umbrella today?"
]

# Embed and query
query_embeddings = model.encode(queries)
query_embeddings = np.array(query_embeddings).astype('float32')

print("\n--- FAISS Similarity Search Results ---")
for i, query in enumerate(queries):
    print(f"\nQuery: {query}")
    distances, indices = index.search(query_embeddings[i:i+1], k=2)
    for j, idx in enumerate(indices[0]):
        print(f"  Top-{j+1}: {sentences[idx]} (Distance: {distances[0][j]:.4f})")
