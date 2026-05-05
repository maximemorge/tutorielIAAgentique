# src/tutorielIAAgentique/masWithMemoryAndSelfReflect.py
import os
import json
import re
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from ddgs import DDGS
from groq import Groq
from dotenv import load_dotenv
from tutorielIAAgentique.utils import debug_print
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / '.env')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
llm = ChatGroq(model='qwen/qwen3-32b', temperature=0.0)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ══════════════════════════════════════════════════════════════
# Shared state
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    query:               str
    plan:                str
    research:            str
    analysis:            str
    critique:            str
    final_answer:        str
    iteration:           int
    sr_retries:          int
    self_reflect_scores: dict
    messages:            Annotated[List, operator.add]


# ══════════════════════════════════════════════════════════════
# Persistent vector memory
# ══════════════════════════════════════════════════════════════
class AgentMemory:
    def __init__(self):
        self.vectorstore = None

    def store(self, query: str, answer: str, metadata: dict = {}) -> None:
        doc = Document(
            page_content=f"Q: {query}\nA: {answer}",
            metadata={'query': query, **metadata}
        )
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], embeddings)
        else:
            self.vectorstore.add_documents([doc])

    def retrieve(self, query: str, k: int = 3) -> str:
        if self.vectorstore is None:
            return "Aucune mémoire disponible."
        docs = self.vectorstore.similarity_search(query, k=k)
        return '\n\n'.join([d.page_content for d in docs])


memory = AgentMemory()


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

def strip_think(text: str) -> str:
    return _THINK_RE.sub('', text).strip()

def search_web(query: str) -> str:
    results = list(DDGS().text(query, max_results=3))
    return '\n'.join([f"- {r['title']}: {r['body']}" for r in results])

def self_reflect(answer: str, query: str) -> dict:
    prompt = (
        "Évalue la réponse ci-dessous sur 3 critères (score entier de 1 à 5) :\n"
        "- completude : tous les aspects de la question sont-ils traités ?\n"
        "- precision   : les faits sont-ils corrects et sourcés ?\n"
        "- clarte      : la réponse est-elle bien structurée ?\n\n"
        f"Question : {query}\n"
        f"Réponse : {answer[:800]}\n\n"
        "Réponds avec UNIQUEMENT cet objet JSON (aucun texte avant ou après) :\n"
        '{"completude": X, "precision": X, "clarte": X, "commentaire": "..."}'
    )
    response = llm.invoke(prompt)
    raw = strip_think(response.content)  # type: ignore
    json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group(0))
            if all(k in scores for k in ('completude', 'precision', 'clarte')):
                return scores
        except json.JSONDecodeError:
            pass
    return {
        'completude': 3, 'precision': 3, 'clarte': 3,
        'commentaire': f'parse error — raw: {raw[:200]!r}',
    }


# ══════════════════════════════════════════════════════════════
# Agents
# ══════════════════════════════════════════════════════════════
def orchestrator_node(state: AgentState) -> AgentState:
    prompt = (
        "Tu es un orchestrateur. Décompose cette question en tâches "
        "pour deux agents : un Researcher (recherche de faits) et un Analyst "
        "(analyse et raisonnement). Sois concis.\n"
        f"Question : {state['query']}"
    )
    response = llm.invoke(prompt)
    debug_print("ORCHESTRATOR PLAN", strip_think(response.content))  # type: ignore
    return {
        'plan':                strip_think(response.content),  # type: ignore
        'iteration':           0,
        'sr_retries':          0,
        'self_reflect_scores': {},
        'critique':            '',
        'final_answer':        '',
    }


def researcher_node(state: AgentState) -> AgentState:
    past_context = memory.retrieve(state['query'])[:600]
    debug_print("MEMORY CONTEXT", past_context)
    web_results = search_web(state['query'])[:400]
    prompt = (
        "Tu es un agent de recherche avec mémoire.\n\n"
        f"Interactions passées pertinentes :\n{past_context}\n\n"
        f"Nouvelle question : {state['query']}\n"
        f"Plan : {state['plan'][:300]}\n\n"
        "Évite de répéter ce qui a déjà été traité. Complète et approfondis. "
        "Fournis des faits vérifiables et précise tes sources.\n\n"
        f"Résultats web récents :\n{web_results}"
    )
    response = llm.invoke(prompt)
    research = strip_think(response.content)  # type: ignore
    debug_print("RESEARCH", research)
    return {'research': research}


def analyst_node(state: AgentState) -> AgentState:
    prompt = (
        "Tu es un analyste expert. Sur la base des recherches, "
        "fournis une analyse approfondie et des conclusions.\n"
        f"Question : {state['query']}\n"
        f"Recherches : {state['research'][:800]}\n"
        "Identifie les limites et incertitudes."
    )
    response = llm.invoke(prompt)
    analysis = strip_think(response.content)  # type: ignore
    debug_print("ANALYSIS", analysis)
    return {'analysis': analysis}


def critic_node(state: AgentState) -> AgentState:
    prompt = (
        "Tu es un agent critique rigoureux. Évalue :\n"
        "1. La qualité factuelle des recherches\n"
        "2. La rigueur de l'analyse\n"
        "3. Les informations manquantes ou contradictoires\n\n"
        f"Recherches : {state['research'][:800]}\n"
        f"Analyse : {state['analysis'][:800]}\n\n"
        "Réponds par APPROVED si satisfaisant, "
        "ou RETRY suivi d'instructions si des améliorations sont nécessaires."
    )
    response = llm.invoke(prompt)
    critique = strip_think(response.content)  # type: ignore
    debug_print("CRITIC", critique)
    return {'critique': critique, 'iteration': state['iteration'] + 1}


def synthesizer_node(state: AgentState) -> AgentState:
    prompt = (
        "Synthétise une réponse finale claire et complète.\n"
        f"Question : {state['query']}\n"
        f"Recherches : {state['research'][:500]}\n"
        f"Analyse : {state['analysis'][:500]}\n"
        "Formate la réponse avec des sections claires."
    )
    response = llm.invoke(prompt)
    final_answer = strip_think(response.content)  # type: ignore
    debug_print("SYNTHESIS", final_answer)

    # ── Self-reflection ──────────────────────────────────────
    scores     = self_reflect(final_answer, state['query'])
    completude = scores.get('completude', 3)
    precision  = scores.get('precision',  3)
    clarte     = scores.get('clarte',     3)
    mean_score = (completude + precision + clarte) / 3
    debug_print(
        "SELF-REFLECT SCORES",
        f"complétude={completude}/5  précision={precision}/5  "
        f"clarté={clarte}/5  → moyenne={mean_score:.2f}/5\n"
        f"Commentaire : {scores.get('commentaire', '')}"
    )

    sr_retries = state.get('sr_retries', 0)  # type: ignore

    # Score trop bas et quota de retries non atteint → RETRY
    if mean_score < 2 and sr_retries < 1:
        retry_reason = (
            f"RETRY – self-reflection score trop bas ({mean_score:.2f}/5). "
            f"Commentaire : {scores.get('commentaire', 'qualité insuffisante')}. "
            "Merci d'approfondir les recherches et l'analyse."
        )
        debug_print("SELF-REFLECT DECISION", f"Score {mean_score:.2f}/5 < 2 → RETRY")
        return {
            'final_answer':        '',
            'critique':            retry_reason,
            'self_reflect_scores': scores,
            'sr_retries':          sr_retries + 1,
        }

    # Qualité acceptable ou quota atteint → on accepte et on stocke
    if mean_score < 2:
        debug_print("SELF-REFLECT DECISION",
                    f"Score {mean_score:.2f}/5 < 2 mais quota SR atteint → ACCEPTED")
    else:
        debug_print("SELF-REFLECT DECISION", f"Score {mean_score:.2f}/5 ≥ 2 → ACCEPTED")

    memory.store(
        query=state['query'],
        answer=final_answer,
        metadata={"iteration": state['iteration'], "self_reflect_mean": mean_score}
    )
    debug_print("MEMORY STORED", f"Stored answer for: {state['query'][:80]}...")

    # ← critique réinitialisé à APPROVED pour stopper la boucle
    return {
        'final_answer':        final_answer,
        'self_reflect_scores': scores,
        'critique':            'APPROVED',
    }


# ══════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════
def should_retry(state: AgentState) -> str:
    """Routing depuis le critic."""
    if "APPROVED" in state["critique"] or state["iteration"] >= 2:
        return "synthesize"
    return "retry"

def after_synthesizer(state: AgentState) -> str:
    """Routing depuis le synthesizer : END si réponse finale, sinon retry."""
    if state.get('final_answer'):
        return END
    return "retry"


# ══════════════════════════════════════════════════════════════
# Graph
# ══════════════════════════════════════════════════════════════
workflow = StateGraph(AgentState)

workflow.add_node('orchestrator', orchestrator_node)
workflow.add_node('researcher',   researcher_node)
workflow.add_node('analyst',      analyst_node)
workflow.add_node('critic',       critic_node)
workflow.add_node('synthesizer',  synthesizer_node)

workflow.set_entry_point('orchestrator')
workflow.add_edge('orchestrator', 'researcher')
workflow.add_edge('researcher',   'analyst')
workflow.add_edge('analyst',      'critic')
workflow.add_conditional_edges(
    'critic',
    should_retry,
    {'synthesize': 'synthesizer', 'retry': 'researcher'}
)
workflow.add_conditional_edges(
    'synthesizer',
    after_synthesizer,
    {'retry': 'researcher', END: END}
)

app = workflow.compile()


# ══════════════════════════════════════════════════════════════
# Run — 3 requêtes successives sur le même thème
# ══════════════════════════════════════════════════════════════
requetes = [
    "Quels sont les impacts économiques de l'IA générative en France d'ici 2030 ?",
    "Quels secteurs français seront les plus touchés par l'IA générative ?",
    "Comment la France se prépare-t-elle aux transformations sociales liées à l'IA ?",
]

for req in requetes:
    print(f"\n{'=' * 60}\nQuery: {req}\n{'=' * 60}")
    result = app.invoke({
        "query":               req,
        "messages":            [],
        "self_reflect_scores": {},
        "sr_retries":          0,
    })
    print(result['final_answer'])
    scores = result.get('self_reflect_scores', {})
    if scores:
        mean = (scores.get('completude', 0) +
                scores.get('precision',  0) +
                scores.get('clarte',     0)) / 3
        print(
            f"\n[Self-reflection finale] "
            f"complétude={scores.get('completude')}  "
            f"précision={scores.get('precision')}  "
            f"clarté={scores.get('clarte')}  "
            f"→ moyenne={mean:.2f}/5"
        )

app.get_graph().draw_mermaid_png(output_file_path='graphWithMemoryAndSelfReflect.png')