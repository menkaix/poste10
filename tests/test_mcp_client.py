"""Tests d'intégration du client MCP vers Backlog Service (Cloud Run).

Ces tests nécessitent :
- Application Default Credentials Google configurées (gcloud auth application-default login)
- Accès réseau au service Cloud Run
"""
import pytest

from app.services.google_auth import get_identity_token
from app.core.config import settings


def test_identity_token_is_fetched():
    """Vérifie qu'on peut obtenir un identity token Google."""
    token = get_identity_token(settings.backlog_service_url)
    assert isinstance(token, str)
    assert len(token) > 0
    # Un JWT a 3 parties séparées par des points
    assert token.count(".") == 2


@pytest.mark.asyncio
async def test_list_tools():
    """Vérifie que le serveur MCP répond et expose des outils."""
    from app.services.mcp_client import list_tools

    tools = await list_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
    print(f"\nOutils disponibles ({len(tools)}) :")
    for t in tools:
        print(f"  - {t['name']}: {t['description']}")
