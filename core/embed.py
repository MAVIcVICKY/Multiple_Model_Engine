"""
Embedding utilities using Google Gemini Embedding 2.

Provides functions to embed text, image files, and raw image bytes
into 3072-dimensional vectors using the gemini-embedding-2 model.

Uses lazy initialization so Django startup doesn't fail
when API keys aren't configured yet.
"""

import os
import google.generativeai as genai

MODEL_NAME = "models/gemini-embedding-2-preview"

_configured = False


def _ensure_configured():
    """Lazily configure the Gemini API on first use."""
    global _configured
    if not _configured:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise Exception(
                "GEMINI_API_KEY is not set. "
                "Please add it to your .env file."
            )
        genai.configure(api_key=api_key)
        _configured = True


def embed_text(text: str) -> list:
    """
    Embed a text string into a vector using Gemini Embedding 2.

    Args:
        text: The text string to embed.

    Returns:
        A list of floats representing the embedding vector (3072 dimensions).
    """
    _ensure_configured()
    result = genai.embed_content(
        model=MODEL_NAME,
        content=text,
    )
    return result["embedding"]


def embed_image(image_path: str) -> list:
    """
    Embed an image file into a vector using Gemini Embedding 2.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        A list of floats representing the embedding vector (3072 dimensions).
    """
    _ensure_configured()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes,
    }

    result = genai.embed_content(
        model=MODEL_NAME,
        content=image_part,
    )
    return result["embedding"]


def embed_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> list:
    """
    Embed raw image bytes into a vector using Gemini Embedding 2.

    Args:
        image_bytes: Raw bytes of the image.
        mime_type: MIME type of the image (default: "image/jpeg").

    Returns:
        A list of floats representing the embedding vector (3072 dimensions).
    """
    _ensure_configured()
    image_part = {
        "mime_type": mime_type,
        "data": image_bytes,
    }

    result = genai.embed_content(
        model=MODEL_NAME,
        content=image_part,
    )
    return result["embedding"]
