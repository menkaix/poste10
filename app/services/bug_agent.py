"""Agent LangChain pour le triage des emails en rapports de bugs.

Utilise les outils MCP du Backlog Service pour créer ou mettre à jour des issues.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from mcp import ClientSession
from pydantic import create_model

from app.core.config import settings
from app.services.email_reader import EmailMessage

# Outils MCP exposés à l'agent (subset pertinent pour les issues)
_ISSUE_TOOLS = {
    "find-issues",
    "find-open-issues",
    "find-critical-issues",
    "find-issue-by-id",
    "find-issues-by-status",
    "find-issues-by-type",
    "find-issues-by-severity",
    "find-issues-by-component",
    "find-issue-by-tracking-reference",
    "create-issue",
    "update-issue",
    "update-issue-status",
}

_SYSTEM_PROMPT = """Tu es un agent de triage d'emails spécialisé dans la détection et la gestion des rapports de bugs.

**Tâche :** Analyse l'email fourni et détermine s'il constitue un rapport de bug.

**Critères pour un rapport de bug :**
- Description d'un dysfonctionnement, comportement inattendu ou erreur
- Mention d'un crash, d'une exception, d'un comportement incorrect
- Retour utilisateur signalant que quelque chose ne fonctionne pas

**Actions à effectuer si c'est un rapport de bug :**
1. Utilise `find-issues` avec des mots-clés pertinents pour chercher si un bug similaire existe déjà.
2. Si le bug est **NOUVEAU** : crée une issue avec `create-issue` en passant un `issueJson` complet.
3. Si le bug est **CONNU** : mets à jour l'issue avec `update-issue` en passant un `issueJson` avec l'`id` de l'issue et les nouveaux détails.

**Champs du issueJson pour `create-issue` (extraire depuis l'email si disponible) :**
- `title` *(obligatoire)* : titre court et descriptif du bug
- `type` : toujours `"BUG"` pour un rapport de bug
- `severity` : `CRITICAL` | `HIGH` | `MEDIUM` | `LOW` | `INFO` — déduire de l'urgence/impact
- `reporter` : adresse email de l'expéditeur
- `environment` : environnement mentionné — déduire selon ces règles :
  - Si une URL dans l'email contient `qualif` ou `qualification` → `"qualification"`
  - Si une URL dans l'email contient `dev`, `development` ou `developpement` → `"development"`
  - Si aucune URL mais mention explicite de l'environnement → utiliser la valeur mentionnée
  - Sinon → `"production"` (valeur par défaut si non précisé)
- `platform` : `WEB` | `MOBILE_IOS` | `MOBILE_ANDROID` | `API` | `MICROSERVICE` | `CLOUD_FUNCTION` — si mentionné
- `component` : nom du composant ou microservice concerné, si mentionné
- `affectedVersion` : version affectée, si mentionnée
- `reproductionSteps` : étapes pour reproduire le bug, si décrites
- `expectedBehavior` : comportement attendu, si décrit
- `actualBehavior` : comportement observé / ce qui s'est mal passé

**Champs du issueJson pour `update-issue` :**
- `id` *(obligatoire)* : identifiant de l'issue existante
- `description` : description mise à jour en ajoutant les nouveaux détails de l'email (conserver l'existant + ajouter)

**Format de réponse finale OBLIGATOIRE :**
Termine toujours ta réponse par un bloc JSON exactement de cette forme :
```json
{"is_bug": true, "action": "created", "issue_id": "abc123", "summary": "résumé en une phrase"}
```
Les valeurs possibles pour `action` : "created", "updated", "none".
Si ce n'est pas un bug : `{"is_bug": false, "action": "none", "issue_id": null, "summary": "..."}`.
"""


@dataclass
class BugProcessingResult:
    is_bug: bool
    action: Literal["created", "updated", "none"]
    issue_id: Optional[str]
    summary: str


async def process_email_for_bugs(
    email: EmailMessage,
    session: ClientSession,
    mcp_tool_schemas: list | None = None,
) -> BugProcessingResult:
    if mcp_tool_schemas is None:
        mcp_tool_schemas = await fetch_mcp_tool_schemas(session)
    tools = build_tools_for_session(mcp_tool_schemas, session)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
    )
    llm_with_tools = llm.bind_tools(tools)
    tools_map = {t.name: t for t in tools}

    email_text = (
        f"De : {email.sender}\n"
        f"Date : {email.date}\n"
        f"Sujet : {email.subject}\n\n"
        f"{email.body}"
    )

    messages: list = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Traite cet email :\n\n{email_text}"),
    ]

    # Boucle agent avec tool calling
    while True:
        response: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name in tools_map:
                tool_result = await tools_map[tool_name].ainvoke(tool_args)
            else:
                tool_result = f"Outil inconnu : {tool_name}"

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

    return _parse_agent_result(response.content)


async def fetch_mcp_tool_schemas(session: ClientSession) -> list:
    """Récupère les schémas MCP une seule fois (indépendant de la session)."""
    result = await session.list_tools()
    return [t for t in result.tools if t.name in _ISSUE_TOOLS]


def build_tools_for_session(mcp_tool_schemas: list, session: ClientSession) -> list[StructuredTool]:
    """Crée les LangChain tools liés à la session courante."""
    return [_mcp_tool_to_langchain(t, session) for t in mcp_tool_schemas]


async def _build_langchain_tools(session: ClientSession) -> list[StructuredTool]:
    schemas = await fetch_mcp_tool_schemas(session)
    return build_tools_for_session(schemas, session)


def _mcp_tool_to_langchain(mcp_tool: Any, session: ClientSession) -> StructuredTool:
    schema = mcp_tool.inputSchema or {}
    properties = schema.get("properties", {})

    # Tous les champs sont Optional côté LangChain — la validation des champs
    # obligatoires est déléguée au serveur MCP.
    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        python_type = _json_type_to_python(field_schema.get("type", "string"))
        fields[field_name] = (Optional[python_type], None)

    # Pydantic v2 : create_model avec annotations
    args_schema = create_model(
        f"Args_{mcp_tool.name.replace('-', '_')}",
        **fields,
    )

    tool_name = mcp_tool.name
    tool_session = session

    async def _coroutine(**kwargs: Any) -> str:
        # Retirer les valeurs None pour ne pas polluer l'appel MCP
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

    # Fallback si le JSON n'est pas trouvé
    return BugProcessingResult(
        is_bug=False,
        action="none",
        issue_id=None,
        summary="Impossible de parser la réponse de l'agent",
    )
