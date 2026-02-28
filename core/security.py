import re
from fastapi import HTTPException

# --- Rate limiting ---
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# --- Blocked content patterns ---
BLOCKED_PATTERNS = [
    r'\b(how to make|how to create|how to build|how to synthesize)\b.{0,30}\b(bomb|weapon|explosive|poison|drug|malware|virus)\b',
    r'\b(child|minor|underage).{0,20}\b(nude|naked|sexual|porn)\b',
    r'\b(kill|murder|assassinate)\b.{0,20}\b(person|people|someone|president|minister)\b',
    r'\b(hack|exploit|crack)\b.{0,20}\b(bank|account|password|system)\b',
]

MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3

EXPLICIT_KEYWORDS = [
    "porn", "pornography", "nude", "naked", "sex tape",
    "child abuse", "cp ", "csam", "gore", "snuff",
    "rape", "molest", "pedophile"
]


def validate_query(query: str) -> str:
    """
    Validate and sanitize a query.
    Raises HTTPException if query is invalid or harmful.
    Returns cleaned query.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = query.strip()

    if len(query) < MIN_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail="Query is too short")

    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long. Maximum {MAX_QUERY_LENGTH} characters allowed."
        )

    query_lower = query.lower()

    # Check explicit keywords
    for keyword in EXPLICIT_KEYWORDS:
        if keyword in query_lower:
            raise HTTPException(
                status_code=400,
                detail="This type of content is not allowed on ResearchMind."
            )

    # Check harmful patterns
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail="This query has been flagged as potentially harmful and cannot be processed."
            )

    return query


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent path traversal."""
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    filename = filename.strip('. ')
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return filename