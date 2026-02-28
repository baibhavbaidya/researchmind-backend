from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from core.database import get_db, ChatHistory
from core.graph import run_research_pipeline
from core.firebase_auth import verify_token, get_user_id
from core.security import limiter, validate_query, sanitize_filename
from utils.pdf_parser import parse_pdf, get_pdf_metadata
import json
import os
import shutil

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_DOCUMENTS_PER_USER = 10
MAX_FILE_SIZE_MB = 20


def get_user_upload_dir(user_id: str) -> str:
    upload_dir = os.path.join(BASE_DIR, "uploaded_docs", user_id)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


class QueryRequest(BaseModel):
    query: str
    use_documents: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    agent_logs: List[str]
    status: str


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)

    # Validate filename
    filename = sanitize_filename(file.filename)
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )
    await file.seek(0)

    # Save to user folder
    upload_dir = get_user_upload_dir(user_id)
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = parse_pdf(file_path)
        metadata = get_pdf_metadata(file_path)

        from main import get_user_retriever
        retriever = get_user_retriever(user_id)
        retriever.index_chunks(chunks)
        retriever.vector_store.save()

        return {
            "message": "Document uploaded successfully",
            "filename": filename,
            "chunks_created": len(chunks),
            "page_count": metadata.get("page_count", 0),
            "document_id": None
        }

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query(
    request: Request,
    body: QueryRequest,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)

    # Validate and sanitize query
    clean_query = validate_query(body.query)

    try:
        from main import get_user_retriever
        retriever = get_user_retriever(user_id)
        has_documents = body.use_documents and retriever.is_ready()

        result = run_research_pipeline(
            query=clean_query,
            retriever=retriever if has_documents else None,
            has_documents=has_documents
        )

        history = ChatHistory(
            user_id=user_id,
            query=clean_query,
            answer=result["answer"],
            sources=json.dumps(result["sources"]),
            agent_logs=json.dumps(result["agent_logs"])
        )
        db.add(history)
        db.commit()

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"],
            agent_logs=result["agent_logs"],
            status=result["status"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@router.get("/history")
@limiter.limit("30/minute")
async def get_history(
    request: Request,
    limit: int = 20,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)

    # Cap limit
    limit = min(limit, 50)

    history = db.query(ChatHistory)\
        .filter(ChatHistory.user_id == user_id)\
        .order_by(ChatHistory.created_at.desc())\
        .limit(limit)\
        .all()

    return [
        {
            "id": h.id,
            "query": h.query,
            "answer": h.answer,
            "sources": json.loads(h.sources) if h.sources else [],
            "agent_logs": json.loads(h.agent_logs) if h.agent_logs else [],
            "created_at": h.created_at.isoformat()
        }
        for h in history
    ]


@router.delete("/history")
@limiter.limit("5/minute")
async def delete_history(
    request: Request,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)
    try:
        db.query(ChatHistory)\
            .filter(ChatHistory.user_id == user_id)\
            .delete()
        db.commit()
        return {"message": "History cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@router.delete("/account")
@limiter.limit("3/minute")
async def delete_account(
    request: Request,
    db: Session = Depends(get_db),
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)
    try:
        # Delete chat history
        db.query(ChatHistory)\
            .filter(ChatHistory.user_id == user_id)\
            .delete()
        db.commit()

        # Delete vector store and uploaded files
        from main import get_user_retriever, user_retrievers
        if user_id in user_retrievers:
            retriever = user_retrievers[user_id]
            retriever.vector_store.clear()
            del user_retrievers[user_id]

        upload_dir = get_user_upload_dir(user_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)

        user_data_dir = os.path.join(BASE_DIR, "data", "users", user_id)
        if os.path.exists(user_data_dir):
            shutil.rmtree(user_data_dir)

        return {"message": "Account deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete account: {str(e)}")

@router.get("/documents")
@limiter.limit("30/minute")
async def get_documents(
    request: Request,
    token: dict = Depends(verify_token)
):
    return []


@router.delete("/documents")
@limiter.limit("10/minute")
async def clear_documents(
    request: Request,
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)

    try:
        from main import get_user_retriever, user_retrievers
        retriever = get_user_retriever(user_id)

        retriever.vector_store.clear()
        retriever.bm25 = None
        retriever.chunks = []

        if user_id in user_retrievers:
            del user_retrievers[user_id]

        upload_dir = get_user_upload_dir(user_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            os.makedirs(upload_dir, exist_ok=True)

        return {"message": "All documents cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@router.delete("/documents/{filename}")
@limiter.limit("10/minute")
async def delete_single_document(
    request: Request,
    filename: str,
    token: dict = Depends(verify_token)
):
    user_id = get_user_id(token)

    try:
        filename = sanitize_filename(filename)
        upload_dir = get_user_upload_dir(user_id)
        file_path = os.path.join(upload_dir, filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")

        # Remove the file
        os.remove(file_path)

        # Rebuild retriever from remaining files
        from main import get_user_retriever, user_retrievers
        if user_id in user_retrievers:
            del user_retrievers[user_id]

        retriever = get_user_retriever(user_id)
        remaining_files = [
            f for f in os.listdir(upload_dir) if f.endswith(".pdf")
        ]

        if remaining_files:
            from utils.pdf_parser import parse_pdf
            all_chunks = []
            for f in remaining_files:
                chunks = parse_pdf(os.path.join(upload_dir, f))
                all_chunks.extend(chunks)
            retriever.index_chunks(all_chunks)
            retriever.vector_store.save()
        else:
            # No docs left — clear vector store
            retriever.vector_store.clear()
            retriever.bm25 = None
            retriever.chunks = []

        return {"message": f"{filename} removed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

class FollowUpRequest(BaseModel):
    original_query: str
    original_answer: str
    followup_question: str

@router.post("/followup")
@limiter.limit("30/minute")
async def followup(
    request: Request,
    body: FollowUpRequest,
    token: dict = Depends(verify_token)
):
    followup_q = validate_query(body.followup_question)

    prompt = f"""You are an expert research assistant with deep analytical capabilities. A user conducted a research query and received a comprehensive answer. They now have a follow-up question that requires thoughtful, accurate, and detailed response.

Original Research Query: {body.original_query}

Research Answer:
{body.original_answer}

Follow-up Question: {followup_q}

Instructions:
- If the follow-up asks for MORE DETAIL on something from the answer, expand on it thoroughly with specific facts, examples, data points, and nuance
- If the follow-up asks to SIMPLIFY or SUMMARIZE, give a clear concise explanation without dumbing it down
- If the follow-up asks a NEW ANGLE not covered in the original answer, research it deeply and give a comprehensive response
- If the follow-up asks WHY or HOW, go deep into mechanisms, causes, and reasoning
- Always be specific — avoid vague generalities, use concrete examples
- If something is uncertain or debated, say so honestly and present multiple perspectives
- Structure your response clearly — use paragraphs, not bullet points unless listing is genuinely needed
- Maintain the same level of academic rigor as the original research answer
- Never make up facts — if you don't know something with confidence, say so

Respond in a way that makes the user feel they are talking to a world-class researcher who genuinely understands the topic."""
    try:
        from langchain_groq import ChatGroq
        import os
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.4
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")