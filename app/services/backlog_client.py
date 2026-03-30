"""Client REST pour le Backlog Service (authentification via identity token Google)."""
import httpx

from app.core.config import settings
from app.services.google_auth import get_identity_token


def _headers() -> dict:
    token = get_identity_token(settings.backlog_service_url)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _url(path: str) -> str:
    return f"{settings.backlog_service_url}{path}"


_TIMEOUT = httpx.Timeout(60.0)


def get_bugs(page: int = 0, size: int = 20) -> list[dict]:
    """Récupère les issues de type BUG (ouvertes, non-dupliquées)."""
    resp = httpx.get(
        _url("/issue/by-status/OPEN"),
        params={"page": page, "size": size},
        headers=_headers(),
        verify=False,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data if isinstance(data, list) else data.get("content", [])
    return [i for i in items if i.get("type") == "BUG"][:size]


def get_duplicates(size: int = 100) -> list[dict]:
    """Récupère les issues marquées DUPLICATE."""
    resp = httpx.get(
        _url("/issue/by-status/DUPLICATE"),
        params={"size": size},
        headers=_headers(),
        verify=False,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    items = data if isinstance(data, list) else data.get("content", [])
    return items[:size]


def get_issue(issue_id: str) -> dict:
    resp = httpx.get(_url(f"/issue/{issue_id}"), headers=_headers(), verify=False, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def add_comment(issue_id: str, author: str, text: str) -> dict:
    """Ajoute un commentaire à une issue (GET → append → PUT)."""
    issue = get_issue(issue_id)
    comments = issue.get("comments") or []
    from datetime import datetime, timezone
    comments.append({
        "author": author,
        "text": text,
        "createDate": datetime.now(timezone.utc).isoformat(),
    })
    issue["comments"] = comments
    resp = httpx.put(
        _url(f"/issue/{issue_id}"),
        json=issue,
        headers=_headers(),
        verify=False,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def update_issue_status(issue_id: str, status: str) -> dict:
    resp = httpx.patch(
        _url(f"/issue/{issue_id}/status"),
        json={"status": status},
        headers=_headers(),
        verify=False,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def mark_as_duplicate(issue_id: str, duplicate_of_id: str) -> dict:
    """Marque une issue comme doublon et met son statut à DUPLICATE."""
    issue = get_issue(issue_id)
    issue["status"] = "DUPLICATE"
    issue["duplicateOfId"] = duplicate_of_id
    resp = httpx.put(
        _url(f"/issue/{issue_id}"),
        json=issue,
        headers=_headers(),
        verify=False,
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def delete_issue(issue_id: str) -> None:
    resp = httpx.delete(_url(f"/issue/{issue_id}"), headers=_headers(), verify=False, timeout=_TIMEOUT)
    resp.raise_for_status()


def trigger_qdrant_index(issue_id: str) -> None:
    """Demande au backlog service d'indexer l'issue dans Qdrant (collection issue-contexts)."""
    resp = httpx.post(
        _url(f"/api/qdrant/index/issue/{issue_id}"),
        headers=_headers(),
        verify=False,
        timeout=_TIMEOUT,
    )
    # Non bloquant : on ignore si l'endpoint n'est pas disponible
    if resp.status_code not in (200, 202, 204):
        pass
