"""Agent de consolidation — Fusionne plusieurs issues en une seule issue exhaustive.

Analyse les descriptions, métadonnées et commentaires de N issues,
puis crée une nouvelle issue consolidée via l'outil MCP `create-issue`.

Utilise create_react_agent (LangGraph) + response_format, même pattern
que bug_report_agent.
"""
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from mcp import ClientSession
from pydantic import BaseModel

from app.core.config import settings
from app.services.bug_agent import build_tools_for_session, fetch_mcp_tool_schemas

_CREATE_TOOLS = {"create-issue"}

_SYSTEM_PROMPT = """Tu es un agent expert en gestion de bugs logiciels. Tu reçois un ensemble
d'issues (rapports de bugs) qui décrivent le même problème ou des problèmes connexes.

**Ta mission :** Analyser toutes ces issues — leurs descriptions, métadonnées et commentaires —
puis créer une seule issue consolidée via `create-issue` qui réunit toutes les informations
factuelles disponibles.

**Principes :**
- Ne rien inventer : chaque fait dans la nouvelle issue doit provenir d'au moins une des issues source.
- Ne rien omettre : si une information apparaît dans une seule issue, elle doit figurer dans la consolidation.
- Éliminer les redondances : ne pas répéter deux fois le même fait.
- Synthétiser sans appauvrir : la nouvelle issue doit être plus complète que chacune prise séparément.

**Champs à renseigner dans `create-issue` :**

- `title` *(obligatoire)* : titre synthétique qui capture le problème central. S'il y a des variantes
  (endpoints différents, composants différents), les mentionner dans le titre ou en sous-titre.

- `type` : toujours `"BUG"`

- `severity` : retenir la sévérité la plus haute parmi toutes les issues source.
  CRITICAL > HIGH > MEDIUM > LOW > INFO

- `reporter` : lister tous les rapporteurs distincts séparés par une virgule.

- `environment` : lister tous les environnements distincts affectés (production, qualification, etc.)

- `platform` : lister toutes les plateformes distinctes affectées.

- `component` : lister tous les composants distincts affectés.

- `affectedVersion` : lister toutes les versions distinctes mentionnées.

- `reproductionSteps` : union de toutes les étapes de reproduction connues, organisées
  par variante si elles diffèrent.

- `expectedBehavior` : ce qui devrait se passer (synthèse si divergent entre issues).

- `actualBehavior` : union de tous les comportements observés — messages d'erreur exacts,
  codes HTTP, stack traces. Regrouper par type de comportement si plusieurs variantes.

- `description` : description factuelle exhaustive en markdown structuré.
  C'est le cœur de la consolidation. Elle doit être aussi minutieuse que si les données
  venaient d'emails bruts. Organiser ainsi :

  ## Résumé de la consolidation
  - Nombre d'issues fusionnées, IDs sources (lister tous les IDs des issues analysées)
  - Nombre de rapporteurs distincts
  - Période d'observation : date la plus ancienne → date la plus récente

  ## Occurrences et contextes
  Pour chaque issue source, détailler :
  - ID de l'issue source
  - Rapporteur et date de signalement
  - Environnement, plateforme, composant, version affectée
  - Comportement observé exact (message d'erreur, code HTTP, stack trace)
  - Étapes de reproduction
  - Commentaires significatifs reçus sur cette occurrence

  ## Comportements observés (synthèse)
  - Union de tous les messages d'erreur, codes HTTP, stack traces distincts
  - Patterns communs et différences entre occurrences

  ## Étapes de reproduction (synthèse)
  - Séquence commune à toutes les occurrences
  - Variantes si les étapes diffèrent selon le contexte

  ## Environnements et plateformes affectés
  - Tableau récapitulatif : environnement × plateforme × composant × version

  ## Contexte technique consolidé
  - Tous les endpoints / URLs impliqués
  - Tous les codes HTTP retournés
  - Identifiants de requête, trace IDs, correlation IDs mentionnés
  - Payloads ou paramètres de requête mentionnés
  - Fichiers source, fonctions, numéros de ligne si mentionnés

  ## Informations complémentaires
  - Toute autre donnée factuelle issue des descriptions ou commentaires :
    logs additionnels, identifiants utilisateur, données métier, tickets liés, etc.
"""


class _ConsolidationOutput(BaseModel):
    """Résultat structuré après exécution de l'agent de consolidation."""
    issue_id: str
    title: str
    summary: str


@dataclass
class ConsolidationResult:
    issue_id: str
    title: str
    summary: str


def _fmt_issue_full(issue: dict) -> str:
    """Formate une issue complète (avec commentaires) pour le prompt."""
    lines = [f"### Issue `{issue.get('id', 'N/A')}`"]
    fields = [
        ("Titre", "title"),
        ("Sévérité", "severity"),
        ("Statut", "status"),
        ("Rapporteur", "reporter"),
        ("Environnement", "environment"),
        ("Plateforme", "platform"),
        ("Composant", "component"),
        ("Version affectée", "affectedVersion"),
        ("Comportement observé", "actualBehavior"),
        ("Comportement attendu", "expectedBehavior"),
        ("Étapes de reproduction", "reproductionSteps"),
        ("Description", "description"),
    ]
    for label, key in fields:
        val = issue.get(key)
        if val:
            lines.append(f"- **{label} :** {val}")

    comments = issue.get("comments") or []
    if comments:
        lines.append("\n**Commentaires :**")
        for c in comments:
            author = c.get("author", "?")
            date = c.get("createDate", "")
            text = c.get("text", "")
            lines.append(f"  - [{author} — {date}] {text}")

    return "\n".join(lines)


async def consolidate_issues(
    issues: list[dict],
    session: ClientSession,
    mcp_tool_schemas: list | None = None,
) -> ConsolidationResult:
    """Analyse N issues et crée une issue consolidée exhaustive via MCP.

    Args:
        issues: liste des issues complètes (avec commentaires) récupérées depuis le backlog.
        session: session MCP active.
        mcp_tool_schemas: schémas pré-chargés (optionnel).
    """
    if len(issues) < 2:
        raise ValueError("Au moins 2 issues sont nécessaires pour une consolidation.")

    if mcp_tool_schemas is None:
        mcp_tool_schemas = await fetch_mcp_tool_schemas(session, _CREATE_TOOLS)
    tools = build_tools_for_session(mcp_tool_schemas, session)

    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=settings.mistral_api_key,
    )

    graph = create_agent(
        llm,
        tools,
        system_prompt=_SYSTEM_PROMPT,
        response_format=_ConsolidationOutput,
    )

    issues_text = "\n\n---\n\n".join(_fmt_issue_full(issue) for issue in issues)
    ids_list = ", ".join(issue.get("id", "?") for issue in issues)

    human_message = (
        f"Consolide les {len(issues)} issues suivantes (IDs : {ids_list}) "
        f"en une seule issue exhaustive :\n\n{issues_text}"
    )

    result = await graph.ainvoke({
        "messages": [HumanMessage(content=human_message)]
    })

    output: _ConsolidationOutput = result["structured_response"]
    return ConsolidationResult(
        issue_id=output.issue_id,
        title=output.title,
        summary=output.summary,
    )


async def fetch_consolidation_tool_schemas(session: ClientSession) -> list:
    """Récupère les schémas MCP pour l'agent de consolidation."""
    return await fetch_mcp_tool_schemas(session, _CREATE_TOOLS)
