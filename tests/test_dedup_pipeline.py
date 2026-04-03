"""Tests unitaires pour le pipeline de déduplication.

Vérifie les deux comportements clés des changements récents :
1. merge_duplicate_bug : supprime la nouvelle issue (delete_issue) au lieu de la marquer DUPLICATE
2. Pipeline email : action="none" / issue_id=None quand doublon détecté
"""
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, call


# ── Fixtures communes ─────────────────────────────────────────────────────────

_NEW_ISSUE = {
    "id": "new-001",
    "title": "Erreur 500 sur /api/paiement",
    "type": "BUG",
    "severity": "HIGH",
    "status": "OPEN",
    "reporter": "user@example.com",
    "environment": "production",
    "platform": "API",
    "component": "paiement-service",
    "actualBehavior": "HTTP 500 retourné à chaque appel",
    "reproductionSteps": "POST /api/paiement avec payload valide",
    "description": "## Informations de l'occurrence\n- Rapporteur : user@example.com",
}

_ORIGINAL_ISSUE = {
    "id": "orig-099",
    "title": "Erreur 500 sur /api/paiement (original)",
    "type": "BUG",
    "severity": "HIGH",
    "status": "OPEN",
    "reporter": "autre@example.com",
    "environment": "production",
    "platform": "API",
    "component": "paiement-service",
    "actualBehavior": "HTTP 500 retourné à chaque appel",
    "description": "Bug initial signalé il y a 3 jours.",
}


# ── Tests de merge_duplicate_bug ──────────────────────────────────────────────

class TestMergeDuplicateBug:
    """Vérifie que merge_duplicate_bug supprime la nouvelle issue et ajoute un commentaire."""

    @pytest.mark.asyncio
    async def test_delete_called_not_mark_as_duplicate(self):
        """delete_issue doit être appelé sur la nouvelle issue, jamais mark_as_duplicate."""
        with (
            patch("app.services.bug_merge_agent.backlog_client") as mock_client,
            patch(
                "app.services.bug_merge_agent._generate_merge_comment",
                new_callable=AsyncMock,
                return_value="Nouvelle occurrence détectée (similarité 92%).",
            ),
        ):
            mock_client.add_comment.return_value = {}
            mock_client.update_issue_status.return_value = {}
            mock_client.delete_issue.return_value = None

            from app.services.bug_merge_agent import merge_duplicate_bug

            result = await merge_duplicate_bug(_NEW_ISSUE, _ORIGINAL_ISSUE, similarity_score=0.92)

        assert result.action == "merged"
        mock_client.delete_issue.assert_called_once_with("new-001")
        assert not hasattr(mock_client.mark_as_duplicate, "called") or not mock_client.mark_as_duplicate.called

    @pytest.mark.asyncio
    async def test_comment_added_to_original(self):
        """Le commentaire doit être ajouté sur l'issue originale, pas sur la nouvelle."""
        with (
            patch("app.services.bug_merge_agent.backlog_client") as mock_client,
            patch(
                "app.services.bug_merge_agent._generate_merge_comment",
                new_callable=AsyncMock,
                return_value="Commentaire test",
            ),
        ):
            mock_client.add_comment.return_value = {}
            mock_client.update_issue_status.return_value = {}
            mock_client.delete_issue.return_value = None

            from app.services.bug_merge_agent import merge_duplicate_bug

            await merge_duplicate_bug(_NEW_ISSUE, _ORIGINAL_ISSUE, similarity_score=0.95)

        mock_client.add_comment.assert_called_once()
        call_args = mock_client.add_comment.call_args
        assert call_args[0][0] == "orig-099", "Le commentaire doit être sur l'issue originale"
        assert call_args[0][1] == "poste10-bot"
        assert "Commentaire test" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_original_status_updated_when_open(self):
        """L'issue originale OPEN doit passer à TRIAGED."""
        with (
            patch("app.services.bug_merge_agent.backlog_client") as mock_client,
            patch(
                "app.services.bug_merge_agent._generate_merge_comment",
                new_callable=AsyncMock,
                return_value="...",
            ),
        ):
            mock_client.add_comment.return_value = {}
            mock_client.update_issue_status.return_value = {}
            mock_client.delete_issue.return_value = None

            from app.services.bug_merge_agent import merge_duplicate_bug

            await merge_duplicate_bug(_NEW_ISSUE, _ORIGINAL_ISSUE, similarity_score=0.90)

        mock_client.update_issue_status.assert_called_once_with("orig-099", "TRIAGED")

    @pytest.mark.asyncio
    async def test_original_status_not_updated_when_not_open(self):
        """L'issue originale déjà TRIAGED ne doit pas être mise à jour."""
        triaged_original = {**_ORIGINAL_ISSUE, "status": "TRIAGED"}

        with (
            patch("app.services.bug_merge_agent.backlog_client") as mock_client,
            patch(
                "app.services.bug_merge_agent._generate_merge_comment",
                new_callable=AsyncMock,
                return_value="...",
            ),
        ):
            mock_client.add_comment.return_value = {}
            mock_client.delete_issue.return_value = None

            from app.services.bug_merge_agent import merge_duplicate_bug

            await merge_duplicate_bug(_NEW_ISSUE, triaged_original, similarity_score=0.90)

        mock_client.update_issue_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_comment_on_llm_failure(self):
        """En cas d'échec LLM, le commentaire fallback est utilisé et l'issue est quand même supprimée."""
        with (
            patch("app.services.bug_merge_agent.backlog_client") as mock_client,
            patch(
                "app.services.bug_merge_agent._generate_merge_comment",
                new_callable=AsyncMock,
                side_effect=Exception("LLM indisponible"),
            ),
        ):
            mock_client.add_comment.return_value = {}
            mock_client.update_issue_status.return_value = {}
            mock_client.delete_issue.return_value = None

            from app.services.bug_merge_agent import merge_duplicate_bug

            result = await merge_duplicate_bug(_NEW_ISSUE, _ORIGINAL_ISSUE, similarity_score=0.88)

        assert result.action == "merged"
        mock_client.delete_issue.assert_called_once_with("new-001")
        mock_client.add_comment.assert_called_once()
        # Le commentaire fallback mentionne la similarité
        comment_text = mock_client.add_comment.call_args[0][2]
        assert "88%" in comment_text or "0.88" in comment_text or "Nouvelle occurrence" in comment_text

    @pytest.mark.asyncio
    async def test_action_skipped_on_rest_error(self):
        """Si add_comment échoue, action=skipped et delete n'est pas appelé."""
        with (
            patch("app.services.bug_merge_agent.backlog_client") as mock_client,
            patch(
                "app.services.bug_merge_agent._generate_merge_comment",
                new_callable=AsyncMock,
                return_value="commentaire",
            ),
        ):
            mock_client.add_comment.side_effect = Exception("Erreur réseau")

            from app.services.bug_merge_agent import merge_duplicate_bug

            result = await merge_duplicate_bug(_NEW_ISSUE, _ORIGINAL_ISSUE, similarity_score=0.91)

        assert result.action == "skipped"


# ── Tests du pipeline email (emails.py) ───────────────────────────────────────

@dataclass
class _FakeBugReportResult:
    is_bug: bool
    issue_id: str | None
    summary: str


@dataclass
class _FakeBugSearchResult:
    found: bool
    issue_id: str | None
    score: float | None
    source: str | None
    reasoning: str


@dataclass
class _FakeBugMergeResult:
    action: str
    comment: str
    detail: str


class TestEmailPipelineDedup:
    """Vérifie action/issue_id dans la réponse selon le résultat de déduplication."""

    def _make_pipeline_mocks(
        self,
        report_is_bug: bool,
        report_issue_id: str | None,
        search_found: bool,
        search_issue_id: str | None = None,
        merge_action: str = "merged",
    ):
        """Construit les mocks pour le pipeline complet."""
        report = _FakeBugReportResult(
            is_bug=report_is_bug,
            issue_id=report_issue_id,
            summary="Résumé test",
        )
        search = _FakeBugSearchResult(
            found=search_found,
            issue_id=search_issue_id,
            score=0.93 if search_found else None,
            source="qdrant" if search_found else None,
            reasoning="test",
        )
        merge = _FakeBugMergeResult(action=merge_action, comment="commentaire", detail="ok")
        return report, search, merge

    @pytest.mark.asyncio
    async def test_duplicate_detected_action_none_issue_id_none(self):
        """Quand doublon détecté et fusion réussie : action='none', issue_id=None."""
        report, search, merge = self._make_pipeline_mocks(
            report_is_bug=True,
            report_issue_id="new-001",
            search_found=True,
            search_issue_id="orig-099",
            merge_action="merged",
        )

        with (
            patch("app.routers.emails.mcp_session"),
            patch("app.routers.emails.fetch_report_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.fetch_search_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.create_bug_report", new_callable=AsyncMock, return_value=report),
            patch("app.routers.emails.backlog_client") as mock_bc,
            patch("app.routers.emails.search_similar_bug", new_callable=AsyncMock, return_value=search),
            patch("app.routers.emails.merge_duplicate_bug", new_callable=AsyncMock, return_value=merge),
            patch("app.routers.emails.index_issue"),
            patch("app.routers.emails.ImapEmailReader") as mock_reader_cls,
        ):
            mock_bc.get_issue.return_value = _NEW_ISSUE
            mock_reader = MagicMock()
            mock_reader.fetch_unread.return_value = [
                MagicMock(uid="42", subject="Bug test", sender="u@x.com", date="Thu, 3 Apr 2026 10:00:00 +0000", body="...")
            ]
            mock_reader_cls.return_value = mock_reader

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/emails/process?n=1&ignore_age=true")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert item["action"] == "none", f"action attendu 'none', reçu '{item['action']}'"
        assert item["issue_id"] is None, f"issue_id attendu None, reçu '{item['issue_id']}'"
        assert item["dedup_action"] == "merged"
        assert item["dedup_duplicate_of"] == "orig-099"

    @pytest.mark.asyncio
    async def test_unique_bug_action_created_issue_id_set(self):
        """Quand bug unique : action='created', issue_id renseigné."""
        report, search, _ = self._make_pipeline_mocks(
            report_is_bug=True,
            report_issue_id="new-001",
            search_found=False,
        )

        with (
            patch("app.routers.emails.mcp_session"),
            patch("app.routers.emails.fetch_report_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.fetch_search_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.create_bug_report", new_callable=AsyncMock, return_value=report),
            patch("app.routers.emails.backlog_client") as mock_bc,
            patch("app.routers.emails.search_similar_bug", new_callable=AsyncMock, return_value=search),
            patch("app.routers.emails.merge_duplicate_bug", new_callable=AsyncMock),
            patch("app.routers.emails.index_issue") as mock_index,
            patch("app.routers.emails.ImapEmailReader") as mock_reader_cls,
        ):
            mock_bc.get_issue.return_value = _NEW_ISSUE
            mock_reader = MagicMock()
            mock_reader.fetch_unread.return_value = [
                MagicMock(uid="43", subject="Nouveau bug", sender="u@x.com", date="Thu, 3 Apr 2026 10:00:00 +0000", body="...")
            ]
            mock_reader_cls.return_value = mock_reader

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/emails/process?n=1&ignore_age=true")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert item["action"] == "created"
        assert item["issue_id"] == "new-001"
        assert item["dedup_action"] == "indexed"
        mock_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_not_a_bug_action_none(self):
        """Email non bug : action='none', is_bug=False, issue_id=None."""
        report = _FakeBugReportResult(is_bug=False, issue_id=None, summary="Newsletter, pas un bug")

        with (
            patch("app.routers.emails.mcp_session"),
            patch("app.routers.emails.fetch_report_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.fetch_search_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.create_bug_report", new_callable=AsyncMock, return_value=report),
            patch("app.routers.emails.backlog_client"),
            patch("app.routers.emails.search_similar_bug", new_callable=AsyncMock),
            patch("app.routers.emails.ImapEmailReader") as mock_reader_cls,
        ):
            mock_reader = MagicMock()
            mock_reader.fetch_unread.return_value = [
                MagicMock(uid="44", subject="Newsletter", sender="news@x.com", date="Thu, 3 Apr 2026 10:00:00 +0000", body="...")
            ]
            mock_reader_cls.return_value = mock_reader

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/emails/process?n=1&ignore_age=true")

        assert resp.status_code == 200
        item = resp.json()[0]
        assert item["is_bug"] is False
        assert item["action"] == "none"
        assert item["issue_id"] is None

    @pytest.mark.asyncio
    async def test_orphan_qdrant_entry_cleaned_on_404(self):
        """Si l'issue Qdrant n'existe plus dans le backlog (404), nettoyer Qdrant et traiter comme bug unique."""
        import httpx
        report = _FakeBugReportResult(is_bug=True, issue_id="new-001", summary="Bug")
        search = _FakeBugSearchResult(found=True, issue_id="orphan-999", score=0.91, source="qdrant", reasoning="")

        http_404 = httpx.HTTPStatusError(
            "404 Not Found",
            request=httpx.Request("GET", "http://test/issue/orphan-999"),
            response=httpx.Response(404),
        )

        with (
            patch("app.routers.emails.mcp_session"),
            patch("app.routers.emails.fetch_report_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.fetch_search_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.create_bug_report", new_callable=AsyncMock, return_value=report),
            patch("app.routers.emails.backlog_client") as mock_bc,
            patch("app.routers.emails.search_similar_bug", new_callable=AsyncMock, return_value=search),
            patch("app.routers.emails.remove_issue") as mock_remove,
            patch("app.routers.emails.index_issue") as mock_index,
            patch("app.routers.emails.ImapEmailReader") as mock_reader_cls,
        ):
            # Premier get_issue (nouvelle issue) OK, second (orphan) → 404
            mock_bc.get_issue.side_effect = [_NEW_ISSUE, http_404]
            mock_reader = MagicMock()
            mock_reader.fetch_unread.return_value = [
                MagicMock(uid="46", subject="Bug", sender="u@x.com", date="Thu, 3 Apr 2026 10:00:00 +0000", body="...")
            ]
            mock_reader_cls.return_value = mock_reader

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/emails/process?n=1&ignore_age=true")

        item = resp.json()[0]
        # Entrée Qdrant orpheline nettoyée
        mock_remove.assert_called_once_with("orphan-999")
        # Nouvelle issue indexée et conservée
        mock_index.assert_called_once()
        assert item["action"] == "created"
        assert item["issue_id"] == "new-001"
        assert "orphan" in item["dedup_action"]

    @pytest.mark.asyncio
    async def test_merge_error_issue_kept(self):
        """Si merge_duplicate_bug lève une exception, issue_kept=True (suppression échouée)."""
        report = _FakeBugReportResult(is_bug=True, issue_id="new-001", summary="Bug")
        search = _FakeBugSearchResult(found=True, issue_id="orig-099", score=0.91, source="qdrant", reasoning="")

        with (
            patch("app.routers.emails.mcp_session"),
            patch("app.routers.emails.fetch_report_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.fetch_search_tool_schemas", new_callable=AsyncMock, return_value=[]),
            patch("app.routers.emails.create_bug_report", new_callable=AsyncMock, return_value=report),
            patch("app.routers.emails.backlog_client") as mock_bc,
            patch("app.routers.emails.search_similar_bug", new_callable=AsyncMock, return_value=search),
            patch("app.routers.emails.merge_duplicate_bug", new_callable=AsyncMock, side_effect=Exception("Erreur REST")),
            patch("app.routers.emails.ImapEmailReader") as mock_reader_cls,
        ):
            mock_bc.get_issue.return_value = _ORIGINAL_ISSUE
            mock_reader = MagicMock()
            mock_reader.fetch_unread.return_value = [
                MagicMock(uid="45", subject="Bug", sender="u@x.com", date="Thu, 3 Apr 2026 10:00:00 +0000", body="...")
            ]
            mock_reader_cls.return_value = mock_reader

            from httpx import AsyncClient, ASGITransport
            from app.main import app

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post("/emails/process?n=1&ignore_age=true")

        item = resp.json()[0]
        # Suppression échouée → l'issue existe encore → action="created", issue_id conservé
        assert item["action"] == "created"
        assert item["issue_id"] == "new-001"
        assert "error" in item["dedup_action"]
