from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.firebase_auth import verify_token, get_user_id
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

ws_router = APIRouter()
executor = ThreadPoolExecutor(max_workers=4)

# --- Per-user rate limiting ---
# Stores {user_id: [timestamp1, timestamp2, ...]}
user_query_times = {}
MAX_QUERIES_PER_MINUTE = 10


def is_rate_limited(user_id: str) -> bool:
    now = time.time()
    window = 60  # 1 minute window

    if user_id not in user_query_times:
        user_query_times[user_id] = []

    # Remove timestamps older than 1 minute
    user_query_times[user_id] = [
        t for t in user_query_times[user_id]
        if now - t < window
    ]

    if len(user_query_times[user_id]) >= MAX_QUERIES_PER_MINUTE:
        return True

    user_query_times[user_id].append(now)
    return False


async def get_user_from_ws_token(token: str) -> str:
    try:
        from firebase_admin import auth
        decoded = auth.verify_id_token(token)
        return decoded.get("uid", "")
    except Exception:
        return ""


async def stream_pipeline(websocket: WebSocket, query: str, has_documents: bool, retriever, user_id: str):

    loop = asyncio.get_event_loop()

    async def send(agent: str, status: str, message: str, data: dict = None):
        payload = {
            "agent": agent,
            "status": status,
            "message": message,
            "data": data or {}
        }
        await websocket.send_text(json.dumps(payload))
        await asyncio.sleep(0.05)

    try:
        print(f"Pipeline starting for query: {query}")
        await send("System", "started", f"Starting research pipeline for: {query}")

        # --- Searcher ---
        await send("Searcher", "thinking", "Searching web and documents for relevant sources...")
        from agents.searcher import SearcherAgent
        searcher = SearcherAgent(retriever=retriever if has_documents else None)
        searcher_output = await loop.run_in_executor(
            executor, lambda: searcher.run(query=query, has_documents=has_documents)
        )
        await send("Searcher", "done",
            f"Found {searcher_output['total_results']} relevant sources",
            {"total_results": searcher_output["total_results"],
             "sources_used": searcher_output["sources_used"]}
        )

        # --- Summarizer ---
        await send("Summarizer", "thinking", "Summarizing each source...")
        from agents.summarizer import SummarizerAgent
        summarizer = SummarizerAgent()
        summarizer_output = await loop.run_in_executor(
            executor, lambda: summarizer.run(searcher_output)
        )
        await send("Summarizer", "done",
            f"Created {summarizer_output['total_summaries']} summaries",
            {"total_summaries": summarizer_output["total_summaries"]}
        )

        # --- Critic ---
        await send("Critic", "thinking", "Reviewing summaries for quality and bias...")
        from agents.critic import CriticAgent
        critic = CriticAgent()
        critic_output = await loop.run_in_executor(
            executor, lambda: critic.run(summarizer_output)
        )
        await send("Critic", "done",
            f"{critic_output['total_reliable']}/{critic_output['total_critiqued']} summaries passed",
            {"total_reliable": critic_output["total_reliable"],
             "total_critiqued": critic_output["total_critiqued"]}
        )

        # --- Fact Checker ---
        await send("FactChecker", "thinking", "Cross-verifying claims across sources...")
        from agents.factchecker import FactCheckerAgent
        factchecker = FactCheckerAgent()
        factchecker_output = await loop.run_in_executor(
            executor, lambda: factchecker.run(critic_output)
        )
        await send("FactChecker", "done",
            f"{len(factchecker_output['verified_claims'])} verified, {len(factchecker_output['disputed_claims'])} disputed",
            {"verified": len(factchecker_output["verified_claims"]),
             "disputed": len(factchecker_output["disputed_claims"])}
        )

        # --- Synthesizer ---
        await send("Synthesizer", "thinking", "Generating final structured answer...")
        from agents.synthesizer import SynthesizerAgent
        synthesizer = SynthesizerAgent()
        final_output = await loop.run_in_executor(
            executor, lambda: synthesizer.run(factchecker_output)
        )
        await send("Synthesizer", "done", "Final answer ready",
            {"answer_length": len(final_output["answer"])}
        )

        # --- Save to database ---
        try:
            from core.database import SessionLocal, ChatHistory
            db = SessionLocal()
            history = ChatHistory(
                user_id=user_id,
                query=query,
                answer=final_output["answer"],
                sources=json.dumps(final_output["sources"]),
                agent_logs=json.dumps([])
            )
            db.add(history)
            db.commit()
            db.close()
        except Exception as e:
            print(f"Failed to save history: {e}")

        # --- Final result ---
        await send("System", "complete", "Research complete!", {
            "answer": final_output["answer"],
            "sources": final_output["sources"],
            "verified_claims": len(factchecker_output["verified_claims"]),
            "disputed_claims": len(factchecker_output["disputed_claims"])
        })

    except Exception as e:
        print(f"Pipeline error: {e}")
        await send("System", "error", f"Pipeline error: {str(e)}")


@ws_router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")

    token = websocket.query_params.get("token", "")
    user_id = await get_user_from_ws_token(token)

    if not user_id:
        await websocket.send_text(json.dumps({
            "agent": "System",
            "status": "error",
            "message": "Unauthorized. Please login again.",
            "data": {}
        }))
        await websocket.close()
        return

    print(f"Authenticated WebSocket for user {user_id[:8]}...")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            query = payload.get("query", "").strip()
            use_documents = payload.get("use_documents", True)

            if not query:
                await websocket.send_text(json.dumps({
                    "agent": "System",
                    "status": "error",
                    "message": "Query cannot be empty",
                    "data": {}
                }))
                continue

            # --- Rate limit check ---
            if is_rate_limited(user_id):
                await websocket.send_text(json.dumps({
                    "agent": "System",
                    "status": "error",
                    "message": f"Rate limit exceeded. Maximum {MAX_QUERIES_PER_MINUTE} queries per minute allowed.",
                    "data": {}
                }))
                continue

            # --- Validate query ---
            try:
                from core.security import validate_query
                query = validate_query(query)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "agent": "System", "status": "error",
                    "message": str(e), "data": {}
                }))
                continue

            from main import get_user_retriever
            retriever = get_user_retriever(user_id)
            has_documents = use_documents and retriever.is_ready()

            await stream_pipeline(websocket, query, has_documents, retriever, user_id)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user {user_id[:8]}...")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "agent": "System",
                "status": "error",
                "message": str(e),
                "data": {}
            }))
        except:
            pass