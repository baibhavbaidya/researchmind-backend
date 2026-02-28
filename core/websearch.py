from tavily import TavilyClient
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using Tavily and return clean results.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False
    )

    results = []

    # Add Tavily's own answer summary if available
    if response.get("answer"):
        results.append({
            "source": "Tavily Summary",
            "url": "",
            "title": "Quick Answer",
            "content": response["answer"],
            "type": "summary"
        })

    # Add individual search results
    for r in response.get("results", []):
        results.append({
            "source": r.get("url", ""),
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
            "type": "web"
        })

    return results


def search_academic(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search academic/research focused content.
    Adds 'research paper' to query for better academic results.
    """
    academic_query = f"{query} research paper study"
    return search_web(academic_query, max_results)