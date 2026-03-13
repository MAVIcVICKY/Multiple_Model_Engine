"""
API views for the Multimodal Search Engine.

Provides endpoints for text-based search, image-based search,
and serving the main chat UI.
"""

import json
import traceback

from django.shortcuts import render
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from .embed import embed_text, embed_image_bytes
from .pinecone_db import search_vectors


class HomeView(View):
    """Serve the main chat UI."""

    def get(self, request):
        return render(request, "index.html")


class TextSearchView(APIView):
    """
    POST /api/search/
    Body: {"query": "dogs playing", "filter": "all"}

    Embeds the query text, searches Pinecone, and returns results
    along with a preview of the first 10 embedding dimensions.
    """

    parser_classes = [JSONParser]

    def post(self, request):
        try:
            query = request.data.get("query", "").strip()
            filter_value = request.data.get("filter", "all")

            if not query:
                return Response(
                    {"error": "Query text is required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Generate embedding for the query text
            vector = embed_text(query)

            # Vector preview: first 10 dimensions
            vector_preview = vector[:10]

            # Search Pinecone
            filter_type = None if filter_value == "all" else filter_value
            matches = search_vectors(vector, top_k=6, filter_type=filter_type)

            # Format results
            results = []
            for match in matches:
                metadata = match.metadata if hasattr(match, "metadata") else {}
                filename = metadata.get("filename", "unknown.jpg")
                results.append(
                    {
                        "score": float(match.score) if hasattr(match, "score") else 0.0,
                        "filename": filename,
                        "caption": metadata.get("caption", ""),
                        "type": metadata.get("type", "image"),
                        "source": metadata.get("source", "unknown"),
                        "image_url": f"/media/images/{filename}",
                    }
                )

            return Response(
                {
                    "results": results,
                    "vector_preview": vector_preview,
                    "query": query,
                }
            )

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": f"Search failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ImageSearchView(APIView):
    """
    POST /api/image-search/
    Body: multipart form with "image" file field.

    Embeds the uploaded image, searches Pinecone for similar images,
    and returns results along with a preview of the first 10 embedding dimensions.
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        try:
            image_file = request.FILES.get("image")

            if not image_file:
                return Response(
                    {"error": "An image file is required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Read image bytes
            image_bytes = image_file.read()

            # Determine mime type
            content_type = image_file.content_type or "image/jpeg"

            # Generate embedding for the image
            vector = embed_image_bytes(image_bytes, mime_type=content_type)

            # Vector preview: first 10 dimensions
            vector_preview = vector[:10]

            # Search Pinecone for similar images
            matches = search_vectors(vector, top_k=6, filter_type="image")

            # Format results
            results = []
            for match in matches:
                metadata = match.metadata if hasattr(match, "metadata") else {}
                filename = metadata.get("filename", "unknown.jpg")
                results.append(
                    {
                        "score": float(match.score) if hasattr(match, "score") else 0.0,
                        "filename": filename,
                        "caption": metadata.get("caption", ""),
                        "type": metadata.get("type", "image"),
                        "source": metadata.get("source", "unknown"),
                        "image_url": f"/media/images/{filename}",
                    }
                )

            return Response(
                {
                    "results": results,
                    "vector_preview": vector_preview,
                }
            )

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": f"Image search failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
