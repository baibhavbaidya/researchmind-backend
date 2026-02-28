import fitz  # PyMuPDF
import os
from typing import List, Dict


def parse_pdf(file_path: str) -> List[Dict]:
    """
    Parse a PDF file and split into chunks.
    Returns a list of chunks with text and metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    full_text = ""

    # Extract text from all pages
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += f"\n[Page {page_num + 1}]\n{text}"

    doc.close()

    # Split into chunks
    chunks = split_into_chunks(full_text, chunk_size=300, overlap=50)
    return chunks


def split_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks by word count.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_index,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end
        })

        chunk_index += 1
        start += chunk_size - overlap  # move forward with overlap

    return chunks


def get_pdf_metadata(file_path: str) -> Dict:
    """
    Extract basic metadata from a PDF.
    """
    doc = fitz.open(file_path)
    metadata = {
        "filename": os.path.basename(file_path),
        "page_count": len(doc),
        "title": doc.metadata.get("title", "Unknown"),
        "author": doc.metadata.get("author", "Unknown"),
    }
    doc.close()
    return metadata