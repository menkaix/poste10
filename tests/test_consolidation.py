"""Tests unitaires pour l'agent de consolidation et l'endpoint /issues/consolidate."""
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.bug_consolidation_agent import _fmt_issue_full


# ── Fixtures ──────────────────────────────────────────────────────────────────

_ISSUE_A = {
    "id": "aaa111",
    "title": "Erreur 500 POST /auth/authenticate",
    "type": "BUG",
    "severity": "HIGH",
    "status": "OPEN",
    "reporter": "user1@example.com",
    "environment": "production",
    "platform": "API",
    "component": "auth-service",
    "affectedVersion": "2.3.1",
    "actualBehavior": "HTTP 500 retourné systématiquement",
    "expectedBehavior": "HTTP 200 avec token JWT",
    "reproductionSteps": "POST /auth/authenticate avec credentials valides",
    "description": "## Informations de l'occurrence\n- Rapporteur : user1@example.com",
    "comments": [
        {"author": "dev1", "createDate": "2026-04-01T10:00:00Z", "text": "Confirmé en production"},
    ],
}

_ISSUE_B = {
    "id": "bbb222",
    "title": "Auth service crash sur /authenticate",
    "type": "BUG",
    "severity": "CRITICAL",
    "status": "OPEN",
    "reporter": "user2@example.com",
    "environment": "qualification",
    "platform": "MOBILE_IOS",
    "component": "auth-service",
    "affectedVersion": "2.3.0",
    "actualBehavior": "HTTP 500 avec message NullPointerException",
    "expectedBehavior": "Authentification réussie",
    "reproductionSteps": "POST /auth/authenticate depuis iPhone 14",
    "description": "## Informations de l'occurrence\n- Rapporteur : user2@example.com",
    "comments": [],
}

_CONSOLIDATED_ISSUE = {
    "id": "cons999",
    "title": "Erreur 500 sur /auth/authenticate (auth-service) — 2 occurrences consolidées",
    "type": "BUG",
    "severity": "CRITICAL",
    "status": "OPEN",
    "reporter": "user1@example.com, user2@example.com",
    "environment": "production, qualification",
    "platform": "API, MOBILE_IOS",
    "component": "auth-service",
    "description": "## Résumé de la consolidation\n- 2 issues fusionnées...",
}


# ── Tests _fmt_issue_full ─────────────────────────────────────────────────────

class TestFmtIssueFull:
    def test_includes_all_metadata(self):
        text = _fmt_issue_full(_ISSUE_A)
        assert "aaa111" in text
        assert "Erreur 500 POST /auth/authenticate" in text
        assert "user1@example.com" in text
        assert "production" in text
        assert "auth-service" in text
        assert "2.3.1" in text

    def test_includes_comments(self):
        text = _fmt_issue_full(_ISSUE_A)
        assert "Confirmé en production" in text
        assert "dev1" in text

    def test_no_comments_section_when_empty(self):
        text = _fmt_issue_full(_ISSUE_B)
        assert "Commentaires" not in text

    def test_missing_fields_not_shown(self):
        minimal = {"id": "x", "title": "Test"}
        text = _fmt_issue_full(minimal)
        assert "x" in text
        assert "Test" in text
        # Les champs absents ne doivent pas apparaître
        assert "Rapporteur" not in text


# ── Tests consolidate_issues ──────────────────────────────────────────────────

@dataclass
class _FakeConsolidationOutput:
    issue_id: str
    title: str
    summary: str


class TestConsolidateIssues:
    @pytest.mark.asyncio
    async def test_raises_if_less_than_two_issues(self):
        from app.services.bug_consolidation_agent import consolidate_issues
        with pytest.raises(ValueError, match="Au moins 2 issues"):
            await consolidate_issues([_ISSUE_A], session=MagicMock())

    @pytest.mark.asyncio
    async def test_calls_create_issue_via_agent(self):
        fake_output = _FakeConsolidationOutput(
            issue_id="cons999",
            title="Issue consolidée",
            summary="2 issues fusionnées",
        )

        with (
            patch("app.services.bug_consolidation_agent.fetch_mcp_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.services.bug_consolidation_agent.build_tools_for_session", return_value=[]),
            patch("app.services.bug_consolidation_agent.create_agent") as mock_create_agent,
        ):
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {"structured_response": fake_output}
            mock_create_agent.return_value = mock_graph

            from app.services.bug_consolidation_agent import consolidate_issues
            result = await consolidate_issues([_ISSUE_A, _ISSUE_B], session=MagicMock())

        assert result.issue_id == "cons999"
        assert result.title == "Issue consolidée"
        assert result.summary == "2 issues fusionnées"

    @pytest.mark.asyncio
    async def test_human_message_contains_all_issue_ids(self):
        """Le message envoyé à l'agent doit référencer tous les IDs sources."""
        fake_output = _FakeConsolidationOutput(issue_id="cons999", title="T", summary="S")
        captured_messages = []

        with (
            patch("app.services.bug_consolidation_agent.fetch_mcp_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.services.bug_consolidation_agent.build_tools_for_session", return_value=[]),
            patch("app.services.bug_consolidation_agent.create_agent") as mock_create_agent,
        ):
            async def capture_invoke(payload):
                captured_messages.extend(payload["messages"])
                return {"structured_response": fake_output}

            mock_graph = MagicMock()
            mock_graph.ainvoke = capture_invoke
            mock_create_agent.return_value = mock_graph

            from app.services.bug_consolidation_agent import consolidate_issues
            await consolidate_issues([_ISSUE_A, _ISSUE_B], session=MagicMock())

        assert len(captured_messages) == 1
        content = captured_messages[0].content
        assert "aaa111" in content
        assert "bbb222" in content


# ── Tests endpoint /issues/consolidate ───────────────────────────────────────

@dataclass
class _FakeConsolidationResult:
    issue_id: str
    title: str
    summary: str


class TestConsolidateEndpoint:
    @pytest.mark.asyncio
    async def test_requires_at_least_two_ids(self):
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/issues/consolidate", json={"issue_ids": ["only-one"]})

        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_successful_consolidation(self):
        fake_result = _FakeConsolidationResult(
            issue_id="cons999",
            title="Issue consolidée",
            summary="2 issues fusionnées avec succès",
        )

        with (
            patch("app.routers.issues.backlog_client") as mock_bc,
            patch("app.routers.issues.mcp_session"),
            patch("app.routers.issues.fetch_consolidation_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.issues.consolidate_issues", new_callable=AsyncMock, return_value=fake_result),
            patch("app.routers.issues.index_issue"),
            patch("app.routers.issues.remove_issue"),
        ):
            mock_bc.get_issue.side_effect = [_ISSUE_A, _ISSUE_B, _CONSOLIDATED_ISSUE]
            mock_bc.delete_issue.return_value = None

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/issues/consolidate", json={"issue_ids": ["aaa111", "bbb222"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["new_issue_id"] == "cons999"
        assert data["new_issue_title"] == "Issue consolidée"
        assert data["sources_count"] == 2
        assert set(data["deleted"]) == {"aaa111", "bbb222"}
        assert data["errors"] == []

    @pytest.mark.asyncio
    async def test_sources_deleted_from_backlog_and_qdrant(self):
        fake_result = _FakeConsolidationResult(issue_id="cons999", title="T", summary="S")

        with (
            patch("app.routers.issues.backlog_client") as mock_bc,
            patch("app.routers.issues.mcp_session"),
            patch("app.routers.issues.fetch_consolidation_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.issues.consolidate_issues", new_callable=AsyncMock, return_value=fake_result),
            patch("app.routers.issues.index_issue"),
            patch("app.routers.issues.remove_issue") as mock_remove,
        ):
            mock_bc.get_issue.side_effect = [_ISSUE_A, _ISSUE_B, _CONSOLIDATED_ISSUE]
            mock_bc.delete_issue.return_value = None

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                await ac.post("/issues/consolidate", json={"issue_ids": ["aaa111", "bbb222"]})

        # Suppression backlog
        deleted_ids = {call.args[0] for call in mock_bc.delete_issue.call_args_list}
        assert deleted_ids == {"aaa111", "bbb222"}
        # Suppression Qdrant
        removed_ids = {call.args[0] for call in mock_remove.call_args_list}
        assert removed_ids == {"aaa111", "bbb222"}

    @pytest.mark.asyncio
    async def test_new_issue_indexed_in_qdrant(self):
        fake_result = _FakeConsolidationResult(issue_id="cons999", title="T", summary="S")

        with (
            patch("app.routers.issues.backlog_client") as mock_bc,
            patch("app.routers.issues.mcp_session"),
            patch("app.routers.issues.fetch_consolidation_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.issues.consolidate_issues", new_callable=AsyncMock, return_value=fake_result),
            patch("app.routers.issues.index_issue") as mock_index,
            patch("app.routers.issues.remove_issue"),
        ):
            mock_bc.get_issue.side_effect = [_ISSUE_A, _ISSUE_B, _CONSOLIDATED_ISSUE]
            mock_bc.delete_issue.return_value = None

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                await ac.post("/issues/consolidate", json={"issue_ids": ["aaa111", "bbb222"]})

        mock_index.assert_called_once_with(_CONSOLIDATED_ISSUE)

    @pytest.mark.asyncio
    async def test_partial_fetch_error_still_consolidates_if_enough(self):
        """Si une issue sur 3 est introuvable mais qu'il en reste >= 2, continuer."""
        import httpx as _httpx
        fake_result = _FakeConsolidationResult(issue_id="cons999", title="T", summary="S")

        with (
            patch("app.routers.issues.backlog_client") as mock_bc,
            patch("app.routers.issues.mcp_session"),
            patch("app.routers.issues.fetch_consolidation_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.issues.consolidate_issues", new_callable=AsyncMock, return_value=fake_result),
            patch("app.routers.issues.index_issue"),
            patch("app.routers.issues.remove_issue"),
        ):
            mock_bc.get_issue.side_effect = [
                _ISSUE_A,
                _httpx.HTTPStatusError("404", request=_httpx.Request("GET", "/"), response=_httpx.Response(404)),
                _ISSUE_B,
                _CONSOLIDATED_ISSUE,
            ]
            mock_bc.delete_issue.return_value = None

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/issues/consolidate",
                    json={"issue_ids": ["aaa111", "missing-000", "bbb222"]},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["sources_count"] == 2
        assert len(data["errors"]) == 1
        assert data["errors"][0]["issue_id"] == "missing-000"

    @pytest.mark.asyncio
    async def test_fails_if_less_than_two_issues_fetchable(self):
        """422 si moins de 2 issues récupérables."""
        import httpx as _httpx

        with patch("app.routers.issues.backlog_client") as mock_bc:
            mock_bc.get_issue.side_effect = [
                _httpx.HTTPStatusError("404", request=_httpx.Request("GET", "/"), response=_httpx.Response(404)),
                _httpx.HTTPStatusError("404", request=_httpx.Request("GET", "/"), response=_httpx.Response(404)),
            ]

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/issues/consolidate",
                    json={"issue_ids": ["gone-1", "gone-2"]},
                )

        assert resp.status_code == 422
