"""Agent 3 — Fusion des bugs lors de la déduplication.

Génère un commentaire contextuel et enrichi pour les nouvelles occurrences
d'un bug connu, puis effectue la fusion dans le backlog.

N'utilise pas d'outils — c'est une Chain LangChain pure :
    ChatPromptTemplate | llm
La fusion REST (add_comment, mark_as_duplicate) est orchestrée directement.
"""
from dataclasses import dataclass
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.services import backlog_client

_BOT_AUTHOR = "poste10-bot"

_SYSTEM_PROMPT = """Tu es un agent expert en gestion de bugs logiciels.
Tu rédiges des commentaires de qualité pour documenter les nouvelles occurrences de bugs connus.

Tu reçois les données du bug original et de la nouvelle occurrence.

**Ta mission** : Rédiger un commentaire markdown utile pour l'équipe de développement.

**Ce que le commentaire doit apporter (synthétiser, pas juste répéter) :**
- Signaler la nouvelle occurrence avec le score de confiance
- Mettre en évidence les NOUVELLES informations par rapport à l'issue originale :
  - Nouvel environnement ou plateforme affectée (élargissement de l'impact) ?
  - Différences dans les étapes de reproduction (plus précises, différentes) ?
  - Nouveaux messages d'erreur, codes HTTP ou stack traces ?
  - Fréquence ou impact plus important ?
  - Nouveau composant, endpoint ou version impliqué ?
  - Nouveau rapporteur (le problème touche plusieurs utilisateurs) ?
- Synthétiser les patterns observés si pertinent
- Suggérer des pistes d'investigation si de nouveaux indices apparaissent

**Style :**
- Markdown structuré et concis
- **Gras** pour les informations critiques ou nouvelles
- Ne pas lister bêtement tous les champs — synthétiser et mettre en valeur ce qui est nouveau
- Ton professionnel, orienté résolution

Réponds avec le texte du commentaire uniquement (markdown), sans balise de code, sans introduction.
"""

_HUMAN_TEMPLATE = """\
Score de similarité : {score:.1%}

---
**Bug original (issue existante) :**
{original_summary}

---
**Nouvelle occurrence :**
{new_summary}

---
Rédige le commentaire de fusion."""

_chain_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
])


@dataclass
class BugMergeResult:
    action: Literal["merged", "skipped"]
    comment: str
    detail: str


def _fmt_issue(issue: dict) -> str:
    fields = [
        ("Titre", issue.get("title")),
        ("Rapporteur", issue.get("reporter")),
        ("Environnement", issue.get("environment")),
        ("Plateforme", issue.get("platform")),
        ("Composant", issue.get("component")),
        ("Version", issue.get("affectedVersion")),
        ("Comportement observé", issue.get("actualBehavior")),
        ("Étapes de reproduction", issue.get("reproductionSteps")),
        ("Comportement attendu", issue.get("expectedBehavior")),
        ("Description", issue.get("description")),
    ]
    return "\n".join(f"- **{label} :** {val}" for label, val in fields if val)


async def _generate_merge_comment(new_issue: dict, original_issue: dict, score: float) -> str:
    """Génère le commentaire de fusion via une Chain LangChain (prompt | llm)."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
    )
    chain = _chain_prompt | llm
    response = await chain.ainvoke({
        "score": score,
        "original_summary": _fmt_issue(original_issue),
        "new_summary": _fmt_issue(new_issue),
    })
    return response.content if isinstance(response.content, str) else str(response.content)


async def merge_duplicate_bug(
    new_issue: dict,
    original_issue: dict,
    similarity_score: float,
) -> BugMergeResult:
    """Fusionne un bug doublon avec son original.

    1. Génère un commentaire contextuel (Chain LLM)
    2. Ajoute le commentaire sur l'issue originale (REST)
    3. Met l'original à TRIAGED si OPEN (REST)
    4. Marque le doublon comme DUPLICATE (REST)
    """
    original_id = original_issue.get("id", "")
    new_id = new_issue.get("id", "")

    try:
        comment = await _generate_merge_comment(new_issue, original_issue, similarity_score)
        detail_prefix = ""
    except Exception as e:
        comment = _fallback_comment(new_issue, similarity_score)
        detail_prefix = f"Commentaire LLM indisponible ({e}), fallback utilisé. "

    try:
        backlog_client.add_comment(original_id, _BOT_AUTHOR, comment)
        if original_issue.get("status") == "OPEN":
            backlog_client.update_issue_status(original_id, "TRIAGED")
        backlog_client.mark_as_duplicate(new_id, original_id)
        return BugMergeResult(
            action="merged",
            comment=comment,
            detail=(
                f"{detail_prefix}Fusionné avec #{original_id} "
                f"(similarité {similarity_score:.1%}). "
                "Commentaire ajouté, doublon marqué DUPLICATE."
            ),
        )
    except Exception as e:
        return BugMergeResult(
            action="skipped",
            comment=comment,
            detail=f"{detail_prefix}Erreur lors de la fusion : {e}",
        )


def _fallback_comment(bug: dict, score: float) -> str:
    """Template de secours si le LLM est indisponible."""
    def field(label: str, key: str) -> str | None:
        v = bug.get(key)
        return f"- **{label} :** {v}" if v else None

    meta = [
        field("Titre", "title"),
        field("Rapporteur", "reporter"),
        field("Environnement", "environment"),
        field("Plateforme", "platform"),
        field("Composant", "component"),
        field("Version affectée", "affectedVersion"),
    ]
    lines = [
        f"**Nouvelle occurrence détectée** (similarité {score:.1%})",
        "",
        *[m for m in meta if m],
    ]
    if bug.get("actualBehavior"):
        lines += ["", f"**Comportement observé :** {bug['actualBehavior']}"]
    if bug.get("reproductionSteps"):
        lines += ["", f"**Étapes de reproduction :** {bug['reproductionSteps']}"]
    return "\n".join(lines)
