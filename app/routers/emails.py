from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.config import settings
from app.services.bug_agent import BugProcessingResult, fetch_mcp_tool_schemas, process_email_for_bugs
from app.services.email_reader import EmailMessage, ImapEmailReader
from app.services.mcp_client import mcp_session

router = APIRouter(prefix="/emails", tags=["emails"])

_DEFAULT_MAX_AGE_HOURS = 24


class EmailProcessingResult(BaseModel):
    uid: str
    subject: str
    sender: str
    date: str
    is_bug: bool
    action: Literal["created", "updated", "none"]
    issue_id: Optional[str]
    summary: str
    marked_as_read: bool
    skipped: bool
    skip_reason: Optional[str]


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
    """Récupère les n derniers emails non lus, filtre ceux de plus de `max_age_hours` heures
    (sauf si `ignore_age=true`), puis analyse avec un agent LangChain.
    L'email est marqué comme lu uniquement s'il s'agit d'un rapport de bug.
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

    # Résultats pour les emails ignorés (sans appel à l'agent)
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

    # Charger les schémas MCP une fois pour tous les emails
    async with mcp_session() as init_session:
        schemas = await fetch_mcp_tool_schemas(init_session)

    for email in to_process:
        async with mcp_session() as session:
            agent_result: BugProcessingResult = await process_email_for_bugs(
                email, session, mcp_tool_schemas=schemas
            )

        marked_as_read = False
        if agent_result.is_bug:
            reader.mark_as_read(email.uid)
            marked_as_read = True

        results.append(
            EmailProcessingResult(
                uid=email.uid,
                subject=email.subject,
                sender=email.sender,
                date=email.date,
                is_bug=agent_result.is_bug,
                action=agent_result.action,
                issue_id=agent_result.issue_id,
                summary=agent_result.summary,
                marked_as_read=marked_as_read,
                skipped=False,
                skip_reason=None,
            )
        )

    return results
