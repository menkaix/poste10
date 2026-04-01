"""Agent 1 — Création de rapports de bugs depuis un email.

Extrait toutes les informations pertinentes du bug pour faciliter
son identification, la localisation des sources, et sa correction.
Crée l'issue dans le backlog via l'outil MCP `create-issue`.

Utilise create_react_agent (LangGraph) pour la boucle outil automatique
et response_format pour l'extraction structurée sans parsing regex.
"""
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from mcp import ClientSession
from pydantic import BaseModel

from app.core.config import settings
from app.services.email_reader import EmailMessage
from app.services.bug_agent import build_tools_for_session, fetch_mcp_tool_schemas

_CREATE_TOOLS = {"create-issue"}

_SYSTEM_PROMPT = """Tu es un agent expert en analyse de rapports de bugs logiciels.

**Tâche :** Analyse l'email et détermine s'il constitue un rapport de bug.

**Critères pour un rapport de bug :**
- Description d'un dysfonctionnement, comportement inattendu ou erreur
- Mention d'un crash, d'une exception, d'un comportement incorrect
- Retour utilisateur signalant que quelque chose ne fonctionne pas

**Si c'est un rapport de bug, crée une issue exhaustive avec `create-issue`.**

**Extraction maximale depuis l'email (tout ce qui est disponible) :**
- `title` *(obligatoire)* : titre court et descriptif, inclure le composant si identifiable
- `type` : toujours `"BUG"`
- `severity` : évaluer depuis l'urgence et l'impact décrit
  - CRITICAL : service en panne, perte de données, sécurité compromise
  - HIGH : fonctionnalité majeure bloquée, impact utilisateur important
  - MEDIUM : fonctionnalité dégradée, contournement possible
  - LOW : gêne mineure, cosmétique
  - INFO : observation sans impact fonctionnel
- `reporter` : adresse email de l'expéditeur
- `environment` : déduire depuis les URLs présentes dans l'email
  - URL contient `qualif` ou `qualification` → `"qualification"`
  - URL contient `dev`, `development` ou `developpement` → `"development"`
  - Mention explicite sans URL → utiliser la valeur mentionnée
  - Sinon → `"production"`
- `platform` : WEB | MOBILE_IOS | MOBILE_ANDROID | API | MICROSERVICE | CLOUD_FUNCTION
- `component` : composant ou microservice affecté — extraire même si mentionné implicitement
- `affectedVersion` : version mentionnée dans l'email
- `reproductionSteps` : étapes précises — actions utilisateur, paramètres, contexte de navigation
- `expectedBehavior` : ce qui devrait se passer selon l'utilisateur
- `actualBehavior` : ce qui se passe réellement — messages d'erreur exacts, codes HTTP, stack traces
- `description` : synthèse enrichie en markdown structuré. Inclure tous les éléments disponibles,
  organisés en sections claires. Extraire et mentionner si présents dans l'email :

  **Contexte temporel**
  - Date et heure exacte de l'occurrence (depuis l'email ou le contenu)
  - Fréquence : première fois, récurrent, depuis quand ?

  **Localisation du code**
  - Fichier(s) source concerné(s) et numéro de ligne si mentionnés
  - Fonction, méthode ou classe incriminée
  - Stack trace complète si présente dans l'email

  **Caractéristiques de l'appareil / environnement client**
  - Système d'exploitation (nom, version)
  - Navigateur (nom, version) pour les bugs web
  - Appareil : marque, modèle pour les bugs mobile
  - Version de l'application installée sur l'appareil
  - Résolution / taille d'écran si pertinent

  **Contexte réseau et serveur**
  - URL exacte ou endpoint appelé au moment de l'erreur
  - Code HTTP retourné
  - Identifiant de requête ou trace ID si disponible

  **Indices sur la cause**
  - Patterns observés : conditions déclenchantes, reproductibilité
  - Toute autre information contextuelle pouvant aider à localiser et corriger le bug
"""


class _BugReportOutput(BaseModel):
    """Résultat structuré extrait par le LLM après exécution de l'agent."""
    is_bug: bool
    action: Literal["created", "none"]
    issue_id: Optional[str] = None
    summary: str


@dataclass
class BugReportResult:
    is_bug: bool
    issue_id: Optional[str]
    summary: str


async def create_bug_report(
    email: EmailMessage,
    session: ClientSession,
    mcp_tool_schemas: list | None = None,
) -> BugReportResult:
    """Analyse un email et crée une issue de bug si pertinent.

    La boucle outil (ReAct) est gérée par LangGraph.
    La sortie structurée est extraite via response_format sans parsing regex.
    """
    if mcp_tool_schemas is None:
        mcp_tool_schemas = await fetch_mcp_tool_schemas(session, _CREATE_TOOLS)
    tools = build_tools_for_session(mcp_tool_schemas, session)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
    )

    graph = create_agent(
        llm,
        tools,
        system_prompt=_SYSTEM_PROMPT,
        response_format=_BugReportOutput,
    )

    email_text = (
        f"De : {email.sender}\n"
        f"Date : {email.date}\n"
        f"Sujet : {email.subject}\n\n"
        f"{email.body}"
    )

    result = await graph.ainvoke({
        "messages": [HumanMessage(content=f"Traite cet email :\n\n{email_text}")]
    })

    output: _BugReportOutput = result["structured_response"]
    return BugReportResult(
        is_bug=output.is_bug,
        issue_id=output.issue_id,
        summary=output.summary,
    )


async def fetch_report_tool_schemas(session: ClientSession) -> list:
    """Récupère les schémas MCP pour l'agent de création."""
    return await fetch_mcp_tool_schemas(session, _CREATE_TOOLS)
