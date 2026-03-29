import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services import backlog_client
from app.services.dedup_service import DeduplicationOutcome, deduplicate_issue
from app.services.qdrant_dedup import embed_issue

router = APIRouter(prefix="/issues", tags=["issues"])

_TIMEOUT_SECONDS = 180


class DeduplicationResult(BaseModel):
    issue_id: str
    title: str
    action: str  # "indexed" | "duplicate_merged" | "skipped"
    duplicate_of: Optional[str] = None
    similarity_score: Optional[float] = None
    detail: str


@router.post("/deduplicate", response_model=list[DeduplicationResult])
def deduplicate_bugs(
    n: int = Query(default=10, ge=1, le=100, description="Nombre de bugs à traiter"),
):
    """Pour chaque bug récent :
    1. Cherche dans Qdrant si un bug similaire existe.
    2. Nouveau bug → indexé dans Qdrant.
    3. Bug connu → commentaire ajouté au bug original, doublon marqué DUPLICATE puis supprimé.
    """
    bugs = backlog_client.get_bugs(size=n)
    # Traiter du plus ancien au plus récent : garantit que la première occurrence est préservée
    bugs.sort(key=lambda b: b.get("creationDate") or "")

    # Calcul des embeddings en parallèle (I/O Gemini API)
    with ThreadPoolExecutor() as executor:
        vectors = list(executor.map(embed_issue, bugs))

    results: list[DeduplicationResult] = []
    deadline = time.monotonic() + _TIMEOUT_SECONDS

    # Search + index/merge séquentiel (évite la race condition intra-batch)
    for bug, vector in zip(bugs, vectors):
        if time.monotonic() > deadline:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout: traitement interrompu après {_TIMEOUT_SECONDS // 60} minutes. "
                       f"{len(results)}/{len(bugs)} bugs traités.",
            )
        issue_id = bug.get("id", "")
        title = bug.get("title", "")

        outcome: DeduplicationOutcome = deduplicate_issue(bug, vector=vector)

        results.append(DeduplicationResult(
            issue_id=issue_id,
            title=title,
            action=outcome.action,
            duplicate_of=outcome.duplicate_of,
            similarity_score=outcome.similarity_score,
            detail=outcome.detail,
        ))

    return results
