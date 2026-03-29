import faiss


index = None
documents = []

def create_index(embeddings):
    global index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

def search(query_embedding, k=3):
    if index is None:
        raise RuntimeError("Index has not been built yet. Call build_index() first.")
    distances, indices = index.search(query_embedding, k)

    return [documents[i] for i in indices[0] if i != -1]