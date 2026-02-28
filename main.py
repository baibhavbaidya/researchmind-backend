from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from dotenv import load_dotenv
from core.database import init_db
from core.security import limiter
import os

load_dotenv()

app = FastAPI(
    title="ResearchMind API",
    description="Multi-Agent AI Research Assistant",
    version="1.0.0"
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-user retriever cache
user_retrievers = {}


def get_user_retriever(user_id: str):
    if user_id not in user_retrievers:
        from core.vectorstore import VectorStore
        from core.retriever import HybridRetriever

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        user_data_dir = os.path.join(BASE_DIR, "data", "users", user_id)
        os.makedirs(user_data_dir, exist_ok=True)

        vector_store = VectorStore(user_id=user_id)
        retriever = HybridRetriever(vector_store)

        loaded = retriever.load_existing()
        if loaded:
            print(f"✅ Loaded existing vector store for user {user_id[:8]}...")
        else:
            print(f"📭 New vector store created for user {user_id[:8]}...")

        user_retrievers[user_id] = retriever

    return user_retrievers[user_id]


@app.on_event("startup")
async def startup_event():
    print("🚀 ResearchMind API starting up...")
    init_db()
    print("✅ ResearchMind API ready!")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_users": len(user_retrievers)
    }


from api.routes import router
from api.websocket import ws_router

app.include_router(router)
app.include_router(ws_router)