from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agents.searcher import SearcherAgent
from agents.summarizer import SummarizerAgent
from agents.critic import CriticAgent
from agents.factchecker import FactCheckerAgent
from agents.synthesizer import SynthesizerAgent
from core.retriever import HybridRetriever
from dotenv import load_dotenv
import os

load_dotenv()


class ResearchState(TypedDict):
    query: str
    has_documents: bool
    searcher_output: Dict
    summarizer_output: Dict
    critic_output: Dict
    factchecker_output: Dict
    final_output: Dict
    agent_logs: List[str]
    status: str


def create_research_graph(retriever: HybridRetriever = None):
    """
    Create and return the full multi-agent research pipeline.
    """
    searcher = SearcherAgent(retriever=retriever)
    summarizer = SummarizerAgent()
    critic = CriticAgent()
    factchecker = FactCheckerAgent()
    synthesizer = SynthesizerAgent()

    def run_searcher(state: ResearchState) -> Dict:
        print("\n[Graph] Running Searcher Agent...")
        logs = state.get("agent_logs", [])
        logs.append("Searcher: searching for relevant sources")
        output = searcher.run(
            query=state["query"],
            has_documents=state["has_documents"]
        )
        logs.append(f"Searcher: found {output['total_results']} results")
        return {**state, "searcher_output": output, "agent_logs": logs, "status": "searching"}

    def run_summarizer(state: ResearchState) -> Dict:
        print("\n[Graph] Running Summarizer Agent...")
        logs = state.get("agent_logs", [])
        logs.append("Summarizer: summarizing sources")
        output = summarizer.run(state["searcher_output"])
        logs.append(f"Summarizer: created {output['total_summaries']} summaries")
        return {**state, "summarizer_output": output, "agent_logs": logs, "status": "summarizing"}

    def run_critic(state: ResearchState) -> Dict:
        print("\n[Graph] Running Critic Agent...")
        logs = state.get("agent_logs", [])
        logs.append("Critic: reviewing summary quality")
        output = critic.run(state["summarizer_output"])
        logs.append(f"Critic: {output['total_reliable']}/{output['total_critiqued']} summaries passed")
        return {**state, "critic_output": output, "agent_logs": logs, "status": "critiquing"}

    def run_factchecker(state: ResearchState) -> Dict:
        print("\n[Graph] Running Fact Checker Agent...")
        logs = state.get("agent_logs", [])
        logs.append("FactChecker: verifying claims across sources")
        output = factchecker.run(state["critic_output"])
        logs.append(f"FactChecker: {len(output['verified_claims'])} verified, {len(output['disputed_claims'])} disputed")
        return {**state, "factchecker_output": output, "agent_logs": logs, "status": "factchecking"}

    def run_synthesizer(state: ResearchState) -> Dict:
        print("\n[Graph] Running Synthesizer Agent...")
        logs = state.get("agent_logs", [])
        logs.append("Synthesizer: generating final answer")
        output = synthesizer.run(state["factchecker_output"])
        logs.append("Synthesizer: final answer ready")
        return {**state, "final_output": output, "agent_logs": logs, "status": "complete"}

    graph = StateGraph(ResearchState)
    graph.add_node("searcher", run_searcher)
    graph.add_node("summarizer", run_summarizer)
    graph.add_node("critic", run_critic)
    graph.add_node("factchecker", run_factchecker)
    graph.add_node("synthesizer", run_synthesizer)

    graph.set_entry_point("searcher")
    graph.add_edge("searcher", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_edge("critic", "factchecker")
    graph.add_edge("factchecker", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_research_pipeline(
    query: str,
    retriever: HybridRetriever = None,
    has_documents: bool = False
) -> Dict:
    """
    Run the full research pipeline for a query.
    Returns the complete final output.
    """
    graph = create_research_graph(retriever=retriever)

    initial_state: ResearchState = {
        "query": query,
        "has_documents": has_documents,
        "searcher_output": {},
        "summarizer_output": {},
        "critic_output": {},
        "factchecker_output": {},
        "final_output": {},
        "agent_logs": [],
        "status": "starting"
    }

    print(f"\n{'='*60}")
    print(f"Starting Research Pipeline for: '{query}'")
    print('='*60)

    final_state = graph.invoke(initial_state)

    return {
        "query": query,
        "answer": final_state["final_output"].get("answer", ""),
        "sources": final_state["final_output"].get("sources", []),
        "agent_logs": final_state["agent_logs"],
        "status": final_state["status"]
    }