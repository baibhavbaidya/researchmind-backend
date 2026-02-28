from typing import List, Dict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re
import os

load_dotenv()


class FactCheckerAgent:
    def __init__(self):
        self.name = "FactChecker"
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=600
        )

    def extract_and_verify_claims(self, summaries: List[Dict], query: str) -> List[Dict]:
        """
        Extract key claims and cross verify across sources.
        """
        combined = ""
        for i, s in enumerate(summaries):
            combined += f"\nSource {i+1} ({s['source']}):\n{s['summary']}\n"

        prompt = f"""You are a fact-checker. Given multiple research summaries on the same topic,
extract the 3-5 most important claims and verify them across sources.

Query: {query}

Summaries:
{combined}

You MUST respond in EXACTLY this format, repeating the block for each claim:
CLAIM: write the claim here
STATUS: VERIFIED or DISPUTED or UNVERIFIED
REASON: which sources agree or disagree and why
---
CLAIM: write the claim here
STATUS: VERIFIED or DISPUTED or UNVERIFIED
REASON: which sources agree or disagree and why
---"""

        try:
            response = self.llm.invoke(prompt)
            return self.parse_claims(response.content.strip())
        except Exception as e:
            print(f"  ⚠️ FactChecker error: {e}")
            return []

    def parse_claims(self, response_text: str) -> List[Dict]:
        """
        Parse the structured claims response.
        """
        claims = []
        blocks = re.split(r'---+|\n(?=CLAIM:)', response_text)

        for block in blocks:
            block = block.strip()
            if not block or "CLAIM:" not in block.upper():
                continue

            claim_data = {
                "claim": "",
                "status": "UNVERIFIED",
                "reason": ""
            }

            lines = block.split("\n")
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                if line.upper().startswith("CLAIM:"):
                    claim_data["claim"] = line.split(":", 1)[1].strip()

                elif line.upper().startswith("STATUS:"):
                    status = line.split(":", 1)[1].strip().upper()
                    for s in ["VERIFIED", "DISPUTED", "UNVERIFIED"]:
                        if s in status:
                            claim_data["status"] = s
                            break

                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and not any(
                        lines[j].strip().upper().startswith(k)
                        for k in ["CLAIM:", "STATUS:", "REASON:"]
                    ):
                        reason += " " + lines[j].strip()
                        j += 1
                    claim_data["reason"] = reason

            if claim_data["claim"]:
                claims.append(claim_data)

        return claims

    def run(self, critic_output: Dict) -> Dict:
        """
        Fact check claims across all reliable summaries.
        """
        print(f"\n✔️  Fact Checker Agent: verifying claims across {critic_output['total_reliable']} sources...")

        query = critic_output["query"]
        reliable_summaries = critic_output["reliable_summaries"]

        if not reliable_summaries:
            return {
                "agent": self.name,
                "query": query,
                "claims": [],
                "verified_claims": [],
                "disputed_claims": [],
                "unverified_claims": [],
                "reliable_summaries": [],
                "status": "done"
            }

        claims = self.extract_and_verify_claims(reliable_summaries, query)

        verified = [c for c in claims if c["status"] == "VERIFIED"]
        disputed = [c for c in claims if c["status"] == "DISPUTED"]
        unverified = [c for c in claims if c["status"] == "UNVERIFIED"]

        print(f"  → Found {len(claims)} key claims")
        print(f"  → Verified: {len(verified)} | Disputed: {len(disputed)} | Unverified: {len(unverified)}")

        return {
            "agent": self.name,
            "query": query,
            "claims": claims,
            "verified_claims": verified,
            "disputed_claims": disputed,
            "unverified_claims": unverified,
            "reliable_summaries": reliable_summaries,
            "status": "done"
        }