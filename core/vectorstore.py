import faiss
import numpy as np
import json
import os
from typing import List, Dict
from core.embeddings import embed_text, embed_texts, get_embedding_dimension

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_user_data_dir(user_id: str) -> str:
    """Get the data directory for a specific user."""
    user_dir = os.path.join(BASE_DIR, "data", "users", user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


class VectorStore:
    def __init__(self, user_id: str = "global"):
        self.user_id = user_id
        self.dimension = get_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

        # Per user paths
        user_dir = get_user_data_dir(user_id)
        self.faiss_path = os.path.join(user_dir, "faiss_index.bin")
        self.chunks_path = os.path.join(user_dir, "chunks_metadata.json")

    def add_chunks(self, chunks: List[Dict]):
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        print(f"✅ Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([embed_text(query)], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(1 / (1 + dist))
            results.append(chunk)

        return results

    def save(self):
        faiss.write_index(self.index, self.faiss_path)
        with open(self.chunks_path, "w") as f:
            json.dump(self.chunks, f)
        print(f"✅ Vector store saved for user {self.user_id[:8]}...")

    def load(self) -> bool:
        if os.path.exists(self.faiss_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(self.faiss_path)
            with open(self.chunks_path, "r") as f:
                self.chunks = json.load(f)
            print(f"✅ Vector store loaded — {self.index.ntotal} chunks")
            return True
        return False

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        if os.path.exists(self.faiss_path):
            os.remove(self.faiss_path)
        if os.path.exists(self.chunks_path):
            os.remove(self.chunks_path)
        print(f"✅ Vector store cleared for user {self.user_id[:8]}...")

    def get_total_chunks(self) -> int:
        return self.index.ntotal