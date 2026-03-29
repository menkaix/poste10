"""Tests d'intégration du pipeline email → agent → MCP.

Nécessite :
- IMAP credentials dans .env
- GOOGLE_API_KEY dans .env
- Application Default Credentials Google configurées
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# --- Tests unitaires (parsing) ---

def test_parse_agent_result_new_bug():
    from app.services.bug_agent import _parse_agent_result

    response = """
    Après analyse, cet email décrit bien un bug.
    ```json
    {"is_bug": true, "action": "created", "issue_id": "abc123", "summary": "Crash au login"}
    ```
    """
    result = _parse_agent_result(response)
    assert result.is_bug is True
    assert result.action == "created"
    assert result.issue_id == "abc123"


def test_parse_agent_result_known_bug():
    from app.services.bug_agent import _parse_agent_result

    response = '{"is_bug": true, "action": "updated", "issue_id": "xyz789", "summary": "Bug déjà connu mis à jour"}'
    result = _parse_agent_result(response)
    assert result.is_bug is True
    assert result.action == "updated"
    assert result.issue_id == "xyz789"


def test_parse_agent_result_not_bug():
    from app.services.bug_agent import _parse_agent_result

    response = 'Cet email est une newsletter. {"is_bug": false, "action": "none", "issue_id": null, "summary": "Email non pertinent"}'
    result = _parse_agent_result(response)
    assert result.is_bug is False
    assert result.action == "none"
    assert result.issue_id is None


# --- Test d'intégration : connexion IMAP ---

def test_imap_connection_and_fetch():
    """Vérifie que la connexion IMAP fonctionne et retourne des emails (ou liste vide)."""
    from app.core.config import settings
    from app.services.email_reader import ImapEmailReader

    reader = ImapEmailReader(
        host=settings.imap_host,
        port=settings.imap_port,
        username=settings.imap_username,
        password=settings.imap_password,
    )
    emails = reader.fetch_unread(5)
    assert isinstance(emails, list)
    print(f"\n{len(emails)} email(s) non lu(s) trouvé(s)")
    for e in emails:
        subject = e.subject[:60].encode("ascii", errors="replace").decode()
        sender = e.sender[:40].encode("ascii", errors="replace").decode()
        print(f"  - [{e.uid}] {subject} | De: {sender}")


# --- Test d'intégration complet : endpoint POST /emails/process ---

@pytest.mark.asyncio
async def test_process_emails_endpoint():
    """Appelle l'endpoint avec n=2 et vérifie la structure de la réponse."""
    from httpx import AsyncClient, ASGITransport

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/emails/process?n=2")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print(f"\n{len(data)} email(s) traité(s)")
    for item in data:
        assert "uid" in item
        assert "is_bug" in item
        assert item["action"] in ("created", "updated", "none")
        print(
            f"  - [{item['uid']}] bug={item['is_bug']} "
            f"action={item['action']} | {item['summary'][:80]}"
        )
