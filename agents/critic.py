from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class CriticAgent:
    def __init__(self):
        self.name = "Critic"
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=400
        )

    def critique_summary(self, summary: str, query: str) -> Dict:
        """
        Critique a single summary for quality and reliability.
        """
        prompt = f"""You are a critical research reviewer. Analyze this summary and identify any issues.
Check for:
1. Vague or unsupported claims
2. Potential bias or one-sided perspective
3. Missing important context
4. Contradictions or logical errors
5. Relevance to the query

Query: {query}

Summary: {summary}

You MUST respond in EXACTLY this format with no extra text before or after:
CONFIDENCE: HIGH or MEDIUM or LOW
ISSUES: describe issues here, or write None if no issues found
VERDICT: RELIABLE or QUESTIONABLE or UNRELIABLE"""

        try:
            response = self.llm.invoke(prompt)
            return self.parse_critique(response.content.strip(), summary)
        except Exception as e:
            print(f"  ⚠️ Critic error: {e}")
            return {
                "summary": summary,
                "confidence": "MEDIUM",
                "issues": "Could not critique due to API error",
                "verdict": "RELIABLE",
                "keep": True
            }

    def parse_critique(self, critique_text: str, original_summary: str) -> Dict:
        """
        Parse the structured critique response.
        """
        confidence = "MEDIUM"
        issues = "None"
        verdict = "RELIABLE"

        lines = critique_text.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().upper()
                for level in ["HIGH", "MEDIUM", "LOW"]:
                    if level in confidence:
                        confidence = level
                        break

            elif line.upper().startswith("ISSUES:"):
                issues = line.split(":", 1)[1].strip()
                j = i + 1
                while j < len(lines) and lines[j].strip() and not any(
                    lines[j].strip().upper().startswith(k)
                    for k in ["CONFIDENCE:", "VERDICT:", "ISSUES:"]
                ):
                    issues += " " + lines[j].strip()
                    j += 1
                if not issues:
                    issues = "None"

            elif line.upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
                for v in ["UNRELIABLE", "QUESTIONABLE", "RELIABLE"]:
                    if v in verdict:
                        verdict = v
                        break

        return {
            "summary": original_summary,
            "confidence": confidence,
            "issues": issues,
            "verdict": verdict,
            "keep": verdict != "UNRELIABLE"
        }

    def run(self, summarizer_output: Dict) -> Dict:
        """
        Critique all summaries from the Summarizer Agent.
        """
        print(f"\n🔎 Critic Agent: reviewing {summarizer_output['total_summaries']} summaries...")

        query = summarizer_output["query"]
        summaries = summarizer_output["summaries"]
        critiqued = []

        for i, item in enumerate(summaries):
            print(f"  → Critiquing summary {i+1}/{len(summaries)}...")
            critique = self.critique_summary(item["summary"], query)

            critiqued.append({
                "source_index": item["source_index"],
                "source": item["source"],
                "url": item["url"],
                "type": item["type"],
                "summary": item["summary"],
                "confidence": critique["confidence"],
                "issues": critique["issues"],
                "verdict": critique["verdict"],
                "keep": critique["keep"]
            })

        reliable = [c for c in critiqued if c["keep"]]
        print(f"  → {len(reliable)}/{len(critiqued)} summaries passed critique")

        return {
            "agent": self.name,
            "query": query,
            "critiqued_summaries": critiqued,
            "reliable_summaries": reliable,
            "total_critiqued": len(critiqued),
            "total_reliable": len(reliable),
            "status": "done"
        }