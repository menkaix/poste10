"""Déduplication des bugs via Qdrant (collection `bug-dedup`).

Embedding : text-embedding-004 via Google AI (768 dims, compatible avec
la collection issue-contexts du backlog service).
"""
from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    ScoredPoint,
)

from app.core.config import settings

_COLLECTION = "bug-dedup"
_VECTOR_SIZE = 3072   # gemini-embedding-001
_SIMILARITY_THRESHOLD = 0.88  # au-dessus = doublon probable


_embedding_model_instance: GoogleGenerativeAIEmbeddings | None = None
_qdrant_client_instance: QdrantClient | None = None


def _embedding_model() -> GoogleGenerativeAIEmbeddings:
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.google_api_key,
        )
    return _embedding_model_instance


def _client() -> QdrantClient:
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantClient(
            host=settings.qdrant_host,
            port=6333,
            api_key=settings.qdrant_api_key,
            https=False,
            timeout=30,
            check_compatibility=False,
        )
    return _qdrant_client_instance


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if _COLLECTION not in existing:
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )


def _issue_text(issue: dict) -> str:
    """Construit le texte à embedder depuis une issue."""
    parts = [
        issue.get("title", ""),
        issue.get("description", ""),
        issue.get("component", ""),
        issue.get("platform", ""),
        issue.get("environment", ""),
        issue.get("actualBehavior", ""),
        issue.get("reproductionSteps", ""),
    ]
    return " | ".join(p for p in parts if p)


@dataclass
class DuplicateMatch:
    issue_id: str
    score: float


def embed_issue(issue: dict) -> Optional[list[float]]:
    """Calcule le vecteur d'embedding d'une issue. Retourne None si texte vide."""
    text = _issue_text(issue)
    if not text.strip():
        return None
    return _embedding_model().embed_query(text)


def search_similar(issue: dict, vector: list[float] | None = None) -> Optional[DuplicateMatch]:
    """Cherche un bug similaire dans Qdrant.
    Retourne le meilleur match si score >= seuil, sinon None.
    Accepte un vecteur pré-calculé pour éviter un double appel embedding.
    """
    if vector is None:
        vector = embed_issue(issue)
    if vector is None:
        return None

    client = _client()
    _ensure_collection(client)

    results = client.query_points(
        collection_name=_COLLECTION,
        query=vector,
        limit=1,
        score_threshold=_SIMILARITY_THRESHOLD,
        with_payload=True,
    )

    points = results.points if hasattr(results, "points") else results
    if points:
        best: ScoredPoint = points[0]
        return DuplicateMatch(
            issue_id=best.payload.get("issue_id", ""),
            score=best.score,
        )
    return None


def index_issue(issue: dict, vector: list[float] | None = None) -> None:
    """Indexe un bug dans Qdrant.
    Accepte un vecteur pré-calculé pour éviter un double appel embedding.
    """
    if vector is None:
        vector = embed_issue(issue)
    if vector is None:
        return

    client = _client()
    _ensure_collection(client)

    # Utiliser un hash stable de l'issue_id comme point ID numérique
    issue_id = issue.get("id", "")
    point_id = abs(hash(issue_id)) % (2**63)

    client.upsert(
        collection_name=_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "issue_id": issue_id,
                    "title": issue.get("title", ""),
                    "severity": issue.get("severity", ""),
                    "component": issue.get("component", ""),
                    "status": issue.get("status", ""),
                },
            )
        ],
    )


def remove_issue(issue_id: str) -> None:
    """Retire un bug de l'index Qdrant (après suppression du doublon)."""
    client = _client()
    point_id = abs(hash(issue_id)) % (2**63)
    from qdrant_client.models import PointIdsList
    client.delete(
        collection_name=_COLLECTION,
        points_selector=PointIdsList(points=[point_id]),
    )
