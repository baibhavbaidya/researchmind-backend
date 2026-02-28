from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class SummarizerAgent:
    def __init__(self):
        self.name = "Summarizer"
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=400
        )

    def summarize_single(self, content: str, query: str) -> str:
        """
        Summarize a single source in context of the query.
        """
        prompt = f"""You are a research summarizer. Given a source and a research query,
write a concise 3-4 sentence summary of the source that is relevant to the query.
Focus only on information that helps answer the query.
Be factual and objective.

Query: {query}

Source content:
{content[:2000]}

Write a concise summary:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"  ⚠️ Summarizer error: {e}")
            return content[:300] + "..."

    def run(self, searcher_output: Dict) -> Dict:
        """
        Summarize each result from the Searcher Agent.
        """
        print(f"\n📝 Summarizer Agent: summarizing {searcher_output['total_results']} sources...")

        query = searcher_output["query"]
        results = searcher_output["results"]
        summaries = []

        for i, result in enumerate(results):
            content = result.get("content", "")
            if not content or len(content.strip()) < 50:
                print(f"  → Skipping source {i+1} (too short)")
                continue

            print(f"  → Summarizing source {i+1}/{len(results)}...")
            summary_text = self.summarize_single(content, query)

            summaries.append({
                "source_index": i,
                "source": result.get("source", ""),
                "url": result.get("url", ""),
                "type": result.get("type", "web"),
                "original_content": content[:500],
                "summary": summary_text
            })

        print(f"  → Created {len(summaries)} summaries")

        return {
            "agent": self.name,
            "query": query,
            "summaries": summaries,
            "total_summaries": len(summaries),
            "status": "done"
        }