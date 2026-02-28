from typing import List, Dict
from core.websearch import search_web
from core.retriever import HybridRetriever
from dotenv import load_dotenv
import os

load_dotenv()


class SearcherAgent:
    def __init__(self, retriever: HybridRetriever = None):
        self.retriever = retriever
        self.name = "Searcher"

    def run(self, query: str, has_documents: bool = False) -> Dict:
        """
        Decides where to search based on context.
        - If documents uploaded → search vector store
        - If no documents → search web
        - Always tries both and combines if possible
        """
        print(f"\n🔍 Searcher Agent: processing query: '{query}'")

        results = []
        sources_used = []

        # --- Search uploaded documents if available ---
        if has_documents and self.retriever:
            print("  → Searching uploaded documents...")
            doc_results = self.retriever.search(query, top_k=5)
            if doc_results:
                for r in doc_results:
                    results.append({
                        "content": r["text"],
                        "source": f"Uploaded Document (chunk {r['chunk_id']})",
                        "url": "",
                        "score": r.get("combined_score", r.get("score", 0)),
                        "type": "document"
                    })
                sources_used.append("documents")
                print(f"  → Found {len(doc_results)} chunks from documents")

        # --- Always search web for additional context ---
        print("  → Searching web...")
        web_results = search_web(query, max_results=4)
        if web_results:
            results.extend(web_results)
            sources_used.append("web")
            print(f"  → Found {len(web_results)} web results")

        # Sort all results by score if available
        results = sorted(
            results,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:7]  # keep top 7 total

        return {
            "agent": self.name,
            "query": query,
            "results": results,
            "sources_used": sources_used,
            "total_results": len(results),
            "status": "done"
        }