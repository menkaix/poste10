from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.config import settings
from app.services import backlog_client
from app.services.bug_report_agent import BugReportResult, create_bug_report, fetch_report_tool_schemas
from app.services.bug_search_agent import BugSearchResult, search_similar_bug, fetch_search_tool_schemas
from app.services.bug_merge_agent import BugMergeResult, merge_duplicate_bug
from app.services.email_reader import EmailMessage, ImapEmailReader
from app.services.mcp_client import mcp_session
import httpx

from app.services.qdrant_dedup import index_issue, remove_issue

router = APIRouter(prefix="/emails", tags=["emails"])

_DEFAULT_MAX_AGE_HOURS = 24


class EmailProcessingResult(BaseModel):
    uid: str
    subject: str
    sender: str
    date: str
    is_bug: bool
    action: Literal["created", "none"]
    issue_id: Optional[str]
    summary: str
    marked_as_read: bool
    skipped: bool
    skip_reason: Optional[str]
    dedup_action: Optional[str] = None
    dedup_duplicate_of: Optional[str] = None


def _email_age_hours(email: EmailMessage) -> Optional[float]:
    """Retourne l'âge de l'email en heures, ou None si la date est invalide."""
    try:
        sent_at = parsedate_to_datetime(email.date)
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - sent_at).total_seconds() / 3600
    except Exception:
        return None


@router.post("/process", response_model=list[EmailProcessingResult])
async def process_unread_emails(
    n: int = Query(default=10, ge=1, le=100, description="Nombre d'emails non lus à traiter"),
    max_age_hours: float = Query(
        default=_DEFAULT_MAX_AGE_HOURS,
        ge=0,
        description="Ignorer les emails plus vieux que ce nombre d'heures (0 = pas de limite)",
    ),
    ignore_age: bool = Query(
        default=False,
        description="Si true, traiter tous les emails sans limite d'âge",
    ),
):
    """Traite les emails non lus via un pipeline à 3 agents spécialisés :
    1. Agent de rapport  : détecte si c'est un bug et crée une issue structurée
    2. Agent de recherche : cherche un bug similaire dans Qdrant (RAG) et le backlog (MCP)
    3. Agent de fusion   : si doublon, génère un commentaire contextuel et fusionne
    """
    reader = ImapEmailReader(
        host=settings.imap_host,
        port=settings.imap_port,
        username=settings.imap_username,
        password=settings.imap_password,
    )

    emails = reader.fetch_unread(n)

    # Séparer emails récents / trop vieux
    to_process: list[EmailMessage] = []
    skipped_old: list[tuple[EmailMessage, float]] = []

    for email in emails:
        if ignore_age or max_age_hours == 0:
            to_process.append(email)
            continue
        age = _email_age_hours(email)
        if age is None or age <= max_age_hours:
            to_process.append(email)
        else:
            skipped_old.append((email, age))

    results: list[EmailProcessingResult] = []

    for email, age in skipped_old:
        results.append(
            EmailProcessingResult(
                uid=email.uid,
                subject=email.subject,
                sender=email.sender,
                date=email.date,
                is_bug=False,
                action="none",
                issue_id=None,
                summary="",
                marked_as_read=False,
                skipped=True,
                skip_reason=f"Email trop ancien ({age:.1f}h > {max_age_hours}h)",
            )
        )

    if not to_process:
        return results

    # Pré-charger les schémas MCP une fois pour tous les emails
    async with mcp_session() as init_session:
        report_schemas = await fetch_report_tool_schemas(init_session)
        search_schemas = await fetch_search_tool_schemas(init_session)

    for email in to_process:
        dedup_action: Optional[str] = None
        dedup_duplicate_of: Optional[str] = None
        issue_kept = False  # True si la nouvelle issue est conservée dans le backlog

        # --- Agent 1 : Rapport de bug ---
        try:
            async with mcp_session() as session:
                report: BugReportResult = await create_bug_report(
                    email, session, mcp_tool_schemas=report_schemas
                )
        except BaseException as e:
            # LangGraph peut lever un ExceptionGroup (Python 3.11+) sur erreur LLM
            results.append(EmailProcessingResult(
                uid=email.uid,
                subject=email.subject,
                sender=email.sender,
                date=email.date,
                is_bug=False,
                action="none",
                issue_id=None,
                summary="",
                marked_as_read=False,
                skipped=True,
                skip_reason=f"error agent 1: {type(e).__name__}: {e}",
            ))
            continue

        if report.is_bug and report.issue_id:
            try:
                issue = backlog_client.get_issue(report.issue_id)
            except Exception as e:
                dedup_action = f"error: impossible de récupérer l'issue ({e})"
                issue = None

            if issue:
                # --- Agent 2 : Recherche de doublon ---
                try:
                    async with mcp_session() as session:
                        search_result: BugSearchResult = await search_similar_bug(
                            issue,
                            session,
                            mcp_tool_schemas=search_schemas,
                            exclude_id=report.issue_id,
                        )
                except BaseException as e:
                    dedup_action = f"error agent 2: {type(e).__name__}: {e}"
                    issue_kept = True
                    search_result = None  # type: ignore[assignment]

                if search_result is not None and search_result.found and search_result.issue_id:
                    # --- Agent 3 : Fusion ---
                    # Doublon détecté : ajouter un commentaire sur l'original et supprimer la nouvelle issue
                    try:
                        original_issue = backlog_client.get_issue(search_result.issue_id)
                        merge_result: BugMergeResult = await merge_duplicate_bug(
                            issue,
                            original_issue,
                            similarity_score=search_result.score or 0.0,
                        )
                        dedup_action = merge_result.action
                        dedup_duplicate_of = search_result.issue_id
                        # La nouvelle issue a été supprimée par merge_duplicate_bug
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            # L'issue référencée dans Qdrant n'existe plus dans le backlog :
                            # nettoyer l'entrée orpheline et traiter le bug comme unique
                            try:
                                remove_issue(search_result.issue_id)
                            except Exception:
                                pass
                            try:
                                index_issue(issue)
                                dedup_action = "indexed (orphan nettoyé)"
                                issue_kept = True
                            except Exception as idx_err:
                                dedup_action = f"error: indexation ({idx_err})"
                                issue_kept = True
                        else:
                            dedup_action = f"error: {e}"
                            issue_kept = True
                    except Exception as e:
                        dedup_action = f"error: {e}"
                        issue_kept = True  # Suppression échouée, l'issue existe encore
                elif search_result is not None:
                    # Nouveau bug unique : indexer dans Qdrant pour les futures recherches
                    try:
                        index_issue(issue)
                        dedup_action = "indexed"
                        issue_kept = True
                    except Exception as e:
                        dedup_action = f"error: indexation Qdrant ({e})"
                        issue_kept = True
                # else: search_result is None → Agent 2 a échoué, dedup_action/issue_kept déjà définis

            reader.mark_as_read(email.uid)

        results.append(
            EmailProcessingResult(
                uid=email.uid,
                subject=email.subject,
                sender=email.sender,
                date=email.date,
                is_bug=report.is_bug,
                action="created" if issue_kept else "none",
                issue_id=report.issue_id if issue_kept else None,
                summary=report.summary,
                marked_as_read=report.is_bug,
                skipped=False,
                skip_reason=None,
                dedup_action=dedup_action,
                dedup_duplicate_of=dedup_duplicate_of,
            )
        )

    return results
