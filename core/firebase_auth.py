import firebase_admin
from firebase_admin import credentials, auth
from fastapi import HTTPException, Header
from typing import Optional
import os
import json

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    # Try environment variable first (production)
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        service_account_dict = json.loads(service_account_json)
        cred = credentials.Certificate(service_account_dict)
    else:
        # Fall back to file (local development)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "service-account.json")
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)

    firebase_admin.initialize_app(cred)


async def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization.replace("Bearer ", "")

    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except auth.ExpiredIdTokenError:
        raise HTTPException(status_code=401, detail="Token expired. Please login again.")
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please login again.")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


def get_user_id(decoded_token: dict) -> str:
    return decoded_token.get("uid", "")