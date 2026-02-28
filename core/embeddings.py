import os
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Set HuggingFace token to avoid rate limit warnings
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

MODEL_NAME = "all-MiniLM-L6-v2"
model = None


def get_model() -> SentenceTransformer:
    """
    Load the embedding model (only once).
    """
    global model
    if model is None:
        print(f"Loading embedding model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully!")
    return model


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.
    Returns a list of floats (the vector).
    """
    m = get_model()
    embedding = m.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts in batch (faster than one by one).
    Returns a list of vectors.
    """
    m = get_model()
    embeddings = m.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """
    Returns the dimension of the embedding vector.
    Useful for initializing FAISS index.
    """
    m = get_model()
    return m.get_sentence_embedding_dimension()