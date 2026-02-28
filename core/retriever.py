from rank_bm25 import BM25Okapi
from typing import List, Dict
from core.vectorstore import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.chunks = []

    def index_chunks(self, chunks: List[Dict]):
        """
        Index chunks in both FAISS and BM25.
        """
        self.vector_store.add_chunks(chunks)
        self._build_bm25(chunks)
        print(f"✅ Hybrid retriever indexed {len(chunks)} chunks")

    def _build_bm25(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.
        """
        self.chunks = chunks
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Hybrid search — combines FAISS semantic + BM25 keyword results.
        Deduplicates and re-ranks by combined score.
        """
        results = {}

        # --- FAISS semantic search ---
        faiss_results = self.vector_store.search(query, top_k=top_k)
        for r in faiss_results:
            cid = r["chunk_id"]
            results[cid] = r.copy()
            results[cid]["faiss_score"] = r["score"]
            results[cid]["bm25_score"] = 0.0

        # --- BM25 keyword search ---
        if self.bm25 and self.chunks:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k]

            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

            for idx in top_bm25_indices:
                chunk = self.chunks[idx]
                cid = chunk["chunk_id"]
                normalized_score = bm25_scores[idx] / max_bm25

                if cid in results:
                    results[cid]["bm25_score"] = normalized_score
                else:
                    entry = chunk.copy()
                    entry["faiss_score"] = 0.0
                    entry["bm25_score"] = normalized_score
                    entry["score"] = normalized_score
                    results[cid] = entry

        # --- Combine scores (60% semantic + 40% keyword) ---
        for cid in results:
            results[cid]["combined_score"] = (
                0.6 * results[cid].get("faiss_score", 0) +
                0.4 * results[cid].get("bm25_score", 0)
            )

        sorted_results = sorted(
            results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]

        return sorted_results

    def load_existing(self):
        """
        Load FAISS from disk and rebuild BM25 from loaded chunks.
        Call this on app startup if documents were previously uploaded.
        """
        loaded = self.vector_store.load()
        if loaded and self.vector_store.chunks:
            self._build_bm25(self.vector_store.chunks)
            print(f"✅ Hybrid retriever restored with {len(self.chunks)} chunks")
        return loaded

    def get_total_chunks(self) -> int:
        return self.vector_store.get_total_chunks()

    def is_ready(self) -> bool:
        return self.get_total_chunks() > 0