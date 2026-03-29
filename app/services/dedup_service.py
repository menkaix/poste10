"""Service de déduplication d'une issue unique.

Orchestration Qdrant + Backlog pour une seule issue.
Utilisé depuis le batch endpoint et l'ingestion email.
"""
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from app.services import backlog_client
from app.services.qdrant_dedup import DuplicateMatch, embed_issue, index_issue, remove_issue, search_similar

_BOT_AUTHOR = "poste10-bot"


@dataclass
class DeduplicationOutcome:
    action: Literal["indexed", "duplicate_merged", "skipped"]
    duplicate_of: Optional[str] = None
    similarity_score: Optional[float] = None
    detail: str = ""


def build_duplicate_comment(bug: dict, score: float) -> str:
    def field(label: str, key: str) -> str | None:
        v = bug.get(key)
        return f"- **{label} :** {v}" if v else None

    meta = [
        field("Titre", "title"),
        field("Rapporteur", "reporter"),
        field("Environnement", "environment"),
        field("Plateforme", "platform"),
        field("Composant", "component"),
        field("Version affectée", "affectedVersion"),
    ]

    lines = [
        f"**Nouvelle occurrence détectée** (similarité {score:.1%})",
        "",
        *[m for m in meta if m],
    ]
    if bug.get("actualBehavior"):
        lines += ["", f"**Comportement observé :** {bug['actualBehavior']}"]
    if bug.get("reproductionSteps"):
        lines += ["", f"**Étapes de reproduction :** {bug['reproductionSteps']}"]
    return "\n".join(lines)


def deduplicate_issue(issue: dict, vector: list[float] | None = None) -> DeduplicationOutcome:
    """Déduplique une issue : indexe si nouvelle, fusionne si doublon.

    Args:
        issue: dict complet de l'issue (doit contenir au moins 'id' et 'title').
        vector: vecteur d'embedding pré-calculé (optionnel, calculé si absent).

    Returns:
        DeduplicationOutcome décrivant l'action effectuée.
    """
    issue_id = issue.get("id", "")

    if vector is None:
        vector = embed_issue(issue)

    try:
        match: Optional[DuplicateMatch] = search_similar(issue, vector=vector)
    except Exception as e:
        return DeduplicationOutcome(action="skipped", detail=f"Erreur recherche Qdrant: {e}")

    if match is None:
        try:
            index_issue(issue, vector=vector)
            return DeduplicationOutcome(action="indexed", detail="Indexé dans Qdrant.")
        except Exception as e:
            return DeduplicationOutcome(action="skipped", detail=f"Erreur indexation Qdrant: {e}")

    # Doublon trouvé
    try:
        original = backlog_client.get_issue(match.issue_id)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # L'original n'existe plus — nettoyer Qdrant et indexer comme nouveau
            remove_issue(match.issue_id)
            try:
                index_issue(issue, vector=vector)
            except Exception:
                pass
            return DeduplicationOutcome(
                action="indexed",
                detail=f"Original #{match.issue_id} introuvable (404), indexé comme nouveau.",
            )
        return DeduplicationOutcome(
            action="skipped",
            duplicate_of=match.issue_id,
            similarity_score=round(match.score, 4),
            detail=f"Erreur récupération original: {e}",
        )
    except Exception as e:
        return DeduplicationOutcome(
            action="skipped",
            duplicate_of=match.issue_id,
            similarity_score=round(match.score, 4),
            detail=f"Erreur récupération original: {e}",
        )

    try:
        comment_text = build_duplicate_comment(issue, match.score)
        backlog_client.add_comment(match.issue_id, _BOT_AUTHOR, comment_text)
        if original.get("status") == "OPEN":
            backlog_client.update_issue_status(match.issue_id, "TRIAGED")
        backlog_client.mark_as_duplicate(issue_id, match.issue_id)
        return DeduplicationOutcome(
            action="duplicate_merged",
            duplicate_of=match.issue_id,
            similarity_score=round(match.score, 4),
            detail=(
                f"Fusionné avec #{match.issue_id} "
                f"(similarité {match.score:.1%}). "
                "Commentaire ajouté, marqué DUPLICATE."
            ),
        )
    except Exception as e:
        return DeduplicationOutcome(
            action="skipped",
            duplicate_of=match.issue_id,
            similarity_score=round(match.score, 4),
            detail=f"Erreur lors de la fusion: {e}",
        )
