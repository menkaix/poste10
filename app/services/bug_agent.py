"""Utilitaires partagés pour les agents de traitement des bugs.

Fournit les helpers de conversion MCP → LangChain tools, utilisés
par bug_report_agent, bug_search_agent et bug_merge_agent.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from pydantic import create_model


@dataclass
class BugProcessingResult:
    is_bug: bool
    action: Literal["created", "updated", "none"]
    issue_id: Optional[str]
    summary: str


async def fetch_mcp_tool_schemas(session: ClientSession, tool_names: set[str] | None = None) -> list:
    """Récupère les schémas MCP filtrés par noms d'outils."""
    result = await session.list_tools()
    if tool_names is None:
        return result.tools
    return [t for t in result.tools if t.name in tool_names]


def build_tools_for_session(mcp_tool_schemas: list, session: ClientSession) -> list[StructuredTool]:
    """Crée les LangChain tools liés à la session courante."""
    return [_mcp_tool_to_langchain(t, session) for t in mcp_tool_schemas]


def _mcp_tool_to_langchain(mcp_tool: Any, session: ClientSession) -> StructuredTool:
    schema = mcp_tool.inputSchema or {}
    properties = schema.get("properties", {})

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        python_type = _json_type_to_python(field_schema.get("type", "string"))
        fields[field_name] = (Optional[python_type], None)

    args_schema = create_model(
        f"Args_{mcp_tool.name.replace('-', '_')}",
        **fields,
    )

    tool_name = mcp_tool.name
    tool_session = session

    async def _coroutine(**kwargs: Any) -> str:
        clean_args = {k: v for k, v in kwargs.items() if v is not None}
        call_result = await tool_session.call_tool(tool_name, clean_args)
        parts = []
        for content in call_result.content:
            parts.append(content.text if hasattr(content, "text") else str(content))
        return "\n".join(parts) or "(pas de résultat)"

    return StructuredTool(
        name=tool_name,
        description=mcp_tool.description or tool_name,
        args_schema=args_schema,
        coroutine=_coroutine,
    )


def _json_type_to_python(json_type: str) -> type:
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(json_type, str)


def _parse_agent_result(content: Any) -> BugProcessingResult:
    if isinstance(content, list):
        text = " ".join(
            c.text if hasattr(c, "text") else str(c) for c in content
        )
    else:
        text = str(content)

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r'(\{"is_bug".*?\})', text, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(1))
            return BugProcessingResult(
                is_bug=bool(data.get("is_bug", False)),
                action=data.get("action", "none"),
                issue_id=data.get("issue_id"),
                summary=data.get("summary", ""),
            )
        except json.JSONDecodeError:
            pass

    return BugProcessingResult(
        is_bug=False,
        action="none",
        issue_id=None,
        summary="Impossible de parser la réponse de l'agent",
    )
