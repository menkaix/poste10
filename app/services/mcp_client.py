from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

from app.core.config import settings
from app.services.google_auth import get_identity_token


def _auth_headers() -> dict[str, str]:
    token = get_identity_token(settings.backlog_service_url)
    return {"Authorization": f"Bearer {token}"}


@asynccontextmanager
async def mcp_session():
    """Context manager qui ouvre une session MCP (SSE) authentifiée vers Backlog Service."""
    url = f"{settings.backlog_service_url}/sse"
    headers = _auth_headers()

    async with sse_client(url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools() -> list[dict[str, Any]]:
    async with mcp_session() as session:
        result = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]


async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    async with mcp_session() as session:
        result = await session.call_tool(name, arguments)
        return result.content
