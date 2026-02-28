from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class SynthesizerAgent:
    def __init__(self):
        self.name = "Synthesizer"
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=1000
        )

    def build_context(self, reliable_summaries: List[Dict], claims: List[Dict]) -> str:
        """
        Build a structured context from summaries and verified claims.
        """
        context = "=== VERIFIED RESEARCH SUMMARIES ===\n"
        for i, s in enumerate(reliable_summaries):
            context += f"\n[Source {i+1}] {s['source']}\n"
            context += f"Confidence: {s.get('confidence', 'MEDIUM')}\n"
            context += f"Summary: {s['summary']}\n"

        if claims:
            context += "\n=== FACT-CHECKED CLAIMS ===\n"
            for c in claims:
                status_emoji = "✓" if c["status"] == "VERIFIED" else "⚠" if c["status"] == "DISPUTED" else "?"
                context += f"{status_emoji} [{c['status']}] {c['claim']}\n"

        return context

    def run(self, factchecker_output: Dict) -> Dict:
        """
        Synthesize all agent outputs into a final structured answer.
        """
        print(f"\n✍️  Synthesizer Agent: generating final answer...")

        query = factchecker_output["query"]
        reliable_summaries = factchecker_output.get("reliable_summaries", [])
        all_claims = factchecker_output.get("claims", [])
        disputed_claims = factchecker_output.get("disputed_claims", [])

        context = self.build_context(reliable_summaries, all_claims)

        disputed_warning = ""
        if disputed_claims:
            disputed_warning = f"\nNote: The following claims are disputed across sources: {', '.join([c['claim'] for c in disputed_claims])}"

        prompt = f"""You are a research synthesizer. Based on verified research summaries and fact-checked claims,
write a comprehensive, well-structured answer to the research query.

Research Query: {query}

{context}
{disputed_warning}

Instructions:
- Write a clear, comprehensive answer directly addressing the query
- Cite sources using [Source 1], [Source 2] etc. inline
- Clearly mark any disputed claims with ⚠️
- Structure your answer with these sections:
  ## Summary
  (2-3 sentence overview)
  
  ## Key Findings
  (detailed findings with citations)
  
  ## Disputed or Uncertain Points
  (any conflicting information, or write 'None' if all claims are verified)
  
  ## Conclusion
  (brief conclusion)

Write the answer now:"""

        response = self.llm.invoke(prompt)
        final_answer = response.content.strip()

        # Build sources list for UI
        sources = []
        for i, s in enumerate(reliable_summaries):
            sources.append({
                "index": i + 1,
                "source": s["source"],
                "url": s.get("url", ""),
                "type": s.get("type", "web"),
                "confidence": s.get("confidence", "MEDIUM")
            })

        print(f"  → Final answer generated ({len(final_answer)} characters)")

        return {
            "agent": self.name,
            "query": query,
            "answer": final_answer,
            "sources": sources,
            "total_sources": len(sources),
            "verified_claims_count": len(factchecker_output.get("verified_claims", [])),
            "disputed_claims_count": len(disputed_claims),
            "status": "done"
        }