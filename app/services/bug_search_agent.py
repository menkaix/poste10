"""Agent 2 — Recherche de bugs similaires.

Recherche dans deux sources :
- Base de connaissance RAG (Qdrant) : recherche sémantique par similarité vectorielle
- Base de données backlog : recherche par mots-clés via outils MCP

Utilise create_react_agent (LangGraph) + response_format pour un résultat
structuré sans parsing regex.
"""
from dataclasses import dataclass
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.bug_agent import build_tools_for_session, fetch_mcp_tool_schemas
from app.services.qdrant_dedup import search_similar

_SEARCH_MCP_TOOLS = {
    "find-issues",
    "find-open-issues",
    "find-critical-issues",
    "find-issue-by-id",
    "find-issues-by-status",
    "find-issues-by-type",
    "find-issues-by-severity",
    "find-issues-by-component",
    "find-issue-by-tracking-reference",
}

_SYSTEM_PROMPT = """Tu es un agent expert en recherche de bugs similaires.

Tu reçois les données d'un bug et tu dois déterminer si un bug identique ou très similaire
existe déjà dans la base de données.

**Outils disponibles :**
1. `search-qdrant` : recherche sémantique dans la base RAG Qdrant. Privilégier cet outil en premier.
2. Outils MCP (`find-issues`, `find-issues-by-component`, etc.) : recherche par mots-clés
   pour confirmer ou affiner.

**Stratégie :**
1. Lance `search-qdrant` avec les informations clés du bug.
2. Si score élevé (> 0.88), confirme avec `find-issue-by-id`.
3. Si Qdrant ne trouve rien, affine via les outils MCP (par composant, par mots-clés).
4. Évalue si le bug trouvé correspond vraiment au même problème (même cause probable).
5. Ne jamais retourner l'`exclude_id` comme résultat.
"""


class _QdrantSearchArgs(BaseModel):
    title: str = Field(description="Titre du bug à rechercher")
    component: str = Field(default="", description="Composant ou microservice concerné")
    platform: str = Field(default="", description="Plateforme (WEB, API, etc.)")
    actual_behavior: str = Field(default="", description="Comportement observé / message d'erreur")
    environment: str = Field(default="", description="Environnement (production, qualification, etc.)")


class _BugSearchOutput(BaseModel):
    """Résultat structuré de la recherche de doublon."""
    found: bool
    issue_id: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None  # "qdrant" | "mcp"
    reasoning: str


@dataclass
class BugSearchResult:
    found: bool
    issue_id: Optional[str]
    score: Optional[float]
    source: Optional[str]
    reasoning: str


def _build_qdrant_tool(exclude_id: str | None = None) -> StructuredTool:
    """Construit le tool de recherche Qdrant exposé au LLM."""

    def _search(
        title: str,
        component: str = "",
        platform: str = "",
        actual_behavior: str = "",
        environment: str = "",
    ) -> str:
        issue = {
            "title": title,
            "component": component,
            "platform": platform,
            "actualBehavior": actual_behavior,
            "environment": environment,
        }
        match = search_similar(issue)
        if match is None:
            return "Aucun bug similaire trouvé dans Qdrant."
        if exclude_id and match.issue_id == exclude_id:
            return "Aucun bug similaire trouvé (seul résultat était le bug lui-même)."
        return (
            f"Bug similaire trouvé dans Qdrant : "
            f"ID={match.issue_id}, score={match.score:.1%}"
        )

    return StructuredTool(
        name="search-qdrant",
        description=(
            "Recherche sémantique de bugs similaires dans la base vectorielle Qdrant. "
            "Utiliser en priorité avec les informations clés du bug."
        ),
        args_schema=_QdrantSearchArgs,
        func=_search,
    )


async def search_similar_bug(
    issue: dict,
    session: ClientSession,
    mcp_tool_schemas: list | None = None,
    exclude_id: str | None = None,
) -> BugSearchResult:
    """Recherche un bug similaire dans Qdrant et le backlog via MCP.

    Args:
        issue: dict de l'issue à comparer (doit contenir au moins 'title').
        session: session MCP active.
        mcp_tool_schemas: schémas pré-chargés (optionnel).
        exclude_id: ID à exclure des résultats (évite l'auto-correspondance).
    """
    if mcp_tool_schemas is None:
        mcp_tool_schemas = await fetch_mcp_tool_schemas(session, _SEARCH_MCP_TOOLS)

    mcp_tools = build_tools_for_session(mcp_tool_schemas, session)
    qdrant_tool = _build_qdrant_tool(exclude_id=exclude_id)
    all_tools = [qdrant_tool] + mcp_tools

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
    )

    graph = create_react_agent(
        llm,
        all_tools,
        state_modifier=SystemMessage(content=_SYSTEM_PROMPT),
        response_format=_BugSearchOutput,
    )

    issue_summary = (
        f"Titre : {issue.get('title', '')}\n"
        f"Composant : {issue.get('component', 'non précisé')}\n"
        f"Plateforme : {issue.get('platform', 'non précisée')}\n"
        f"Environnement : {issue.get('environment', 'non précisé')}\n"
        f"Comportement observé : {issue.get('actualBehavior', 'non précisé')}\n"
        f"Étapes de reproduction : {issue.get('reproductionSteps', 'non précisées')}\n"
        f"Description : {issue.get('description', '')}"
    )
    if exclude_id:
        issue_summary += f"\n\n⚠️ Ne pas retourner l'ID {exclude_id} comme résultat."

    result = await graph.ainvoke({
        "messages": [HumanMessage(content=f"Recherche des bugs similaires :\n\n{issue_summary}")]
    })

    output: _BugSearchOutput = result["structured_response"]
    return BugSearchResult(
        found=output.found,
        issue_id=output.issue_id,
        score=output.score,
        source=output.source,
        reasoning=output.reasoning,
    )


async def fetch_search_tool_schemas(session: ClientSession) -> list:
    """Récupère les schémas MCP pour l'agent de recherche."""
    return await fetch_mcp_tool_schemas(session, _SEARCH_MCP_TOOLS)
