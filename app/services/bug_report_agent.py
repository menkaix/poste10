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
- `description` : description factuelle exhaustive en markdown structuré. Recopier fidèlement
  toutes les informations disponibles dans l'email, sans interprétation ni omission.
  Organiser en sections selon ce qui est présent :

  **Informations de l'occurrence**
  - Rapporteur : adresse email de l'expéditeur
  - Date et heure exacte de l'email et/ou de l'occurrence si mentionnée
  - Fréquence : première fois, récurrent, depuis quand, combien de fois ?
  - Environnement : production / qualification / développement
  - Plateforme : WEB, MOBILE_IOS, MOBILE_ANDROID, API, MICROSERVICE, CLOUD_FUNCTION
  - Composant ou microservice affecté
  - Version affectée de l'application

  **Comportement observé**
  - Description exacte de ce qui se passe (reprendre mot pour mot si possible)
  - Messages d'erreur exacts, codes HTTP, codes d'erreur métier
  - Stack trace complète si présente

  **Comportement attendu**
  - Ce que l'utilisateur s'attendait à obtenir

  **Étapes de reproduction**
  - Séquence précise d'actions pour reproduire le bug
  - Paramètres, données, contexte de navigation utilisés

  **Environnement client**
  - Système d'exploitation (nom et version)
  - Navigateur (nom et version) pour les bugs web
  - Appareil (marque, modèle) pour les bugs mobile
  - Version de l'application installée sur l'appareil
  - Résolution / taille d'écran si mentionnée

  **Contexte réseau et serveur**
  - URL exacte ou endpoint appelé au moment de l'erreur
  - Code HTTP retourné
  - Identifiant de requête, trace ID, correlation ID si disponible
  - Payload ou paramètres de la requête si mentionnés

  **Localisation du code**
  - Fichier(s) source concerné(s) et numéro de ligne si mentionnés
  - Fonction, méthode ou classe incriminée

  **Informations complémentaires**
  - Toute autre donnée factuelle présente dans l'email : logs additionnels, captures d'écran
    décrites, tickets liés, identifiants utilisateur, données métier concernées, etc.
  - Ne rien omettre, même si l'information semble mineure
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
