from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Literal, Optional

from app.core.config import settings
from app.services.bug_agent import BugProcessingResult, process_email_for_bugs
from app.services.email_reader import ImapEmailReader
from app.services.mcp_client import mcp_session

router = APIRouter(prefix="/emails", tags=["emails"])


class EmailProcessingResult(BaseModel):
    uid: str
    subject: str
    sender: str
    is_bug: bool
    action: Literal["created", "updated", "none"]
    issue_id: Optional[str]
    summary: str
    marked_as_read: bool


@router.post("/process", response_model=list[EmailProcessingResult])
async def process_unread_emails(
    n: int = Query(default=10, ge=1, le=50, description="Nombre d'emails non lus à traiter"),
):
    """Récupère les n derniers emails non lus, les analyse avec un agent LangChain
    et crée/met à jour les issues MCP en cas de rapport de bug.
    L'email est marqué comme lu uniquement s'il s'agit d'un rapport de bug.
    """
    reader = ImapEmailReader(
        host=settings.imap_host,
        port=settings.imap_port,
        username=settings.imap_username,
        password=settings.imap_password,
    )

    emails = reader.fetch_unread(n)
    results: list[EmailProcessingResult] = []

    async with mcp_session() as session:
        for email in emails:
            agent_result: BugProcessingResult = await process_email_for_bugs(
                email, session
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
                    is_bug=agent_result.is_bug,
                    action=agent_result.action,
                    issue_id=agent_result.issue_id,
                    summary=agent_result.summary,
                    marked_as_read=marked_as_read,
                )
            )

    return results
