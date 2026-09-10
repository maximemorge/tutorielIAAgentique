# src/tutorielIAAgentique/masScore.py
import re
import time
import os
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from ddgs import DDGS
from groq import Groq
from dotenv import load_dotenv
from tutorielIAAgentique.utils import debug_print
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / '.env')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def invoke_with_retry(prompt: str, retries: int = 3, wait: int = 5):
    """Retry call for RateLimitError (429)."""
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if '429' in str(e) and attempt < retries - 1:
                debug_print("RATE LIMIT", f"Pause {wait}s avant retry {attempt+1}/{retries}")
                time.sleep(wait)
            else:
                raise

# Groq (tier on_demand) plafonne la SORTIE à 1000 tokens/requête pour ce modèle :
# on fixe max_tokens sous cette limite, sinon Groq renvoie 429 (OTPM).
llm = ChatGroq(model='qwen/qwen3.8-27b', temperature=0.0, max_tokens=800)

# ── Helper: strip <think>…</think> reasoning blocks ───────────
# Qwen3 emits chain-of-thought tags that must not be treated as content.
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks and collapse extra blank lines."""
    return _THINK_RE.sub('', text).strip()

# ── State shared between all agents ───────────────────────
class AgentState(TypedDict):
    query: str                    # Original question
    plan: str                     # Orchestrator's plan
    research: str                 # Researcher's findings
    analysis: str                 # Analyst's analysis
    critique: str                 # Critic's feedback
    final_answer: str             # Final answer
    iteration: int                # Iteration counter
    messages: Annotated[List, operator.add]  # Message history

# ── Orchestrator Agent ───────────────────────────────────────
def orchestrator_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un orchestrateur. Décompose cette question en tâches
    pour deux agents : un Researcher (recherche de faits) et un Analyst
    (analyse et raisonnement). Sois concis.
    Question : {state["query"]}"""
    response = invoke_with_retry(prompt)
    plan = strip_think(response.content)  # type: ignore
    debug_print("ORCHESTRATOR PLAN", plan)
    return {'plan': plan, 'iteration': 0}


# ── Researcher Agent ──────────────────────────────────────────
def search_web(query: str) -> str:
    """Recherche DuckDuckGo et retourne les 3 premiers résultats."""
    results = list(DDGS().text(query, max_results=3))
    return '\n'.join([f"- {r['title']}: {r['body']}" for r in results])

def researcher_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un agent de recherche. Recherche des informations
    factuelles pour répondre à : {state['query']}
    Plan : {state['plan']}
    Utilise tes connaissances et sois précis avec les sources.
    Fournis des faits vérifiables."""
    # Ici : on peut injecter les outils de la Partie 1
    web_results = search_web(state['query'])
    prompt_with_context = prompt + f"\n\nRésultats web :\n{web_results}"
    response = invoke_with_retry(prompt_with_context)
    research = strip_think(response.content)  # type: ignore
    debug_print("RESEARCH", research)
    return {'research': research}

# ── Analyst Agent ─────────────────────────────────────────────
# Troncature des entrées pour rester sous la limite TPM de Groq.
_MAX_RESEARCH_CHARS = 800

def analyst_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un analyste expert. Sur la base des recherches,
    fournis une analyse approfondie et des conclusions.
    Question : {state['query']}
    Recherches : {state['research'][:_MAX_RESEARCH_CHARS]}
    Identifie les limites et incertitudes."""
    response = invoke_with_retry(prompt)
    analysis = strip_think(response.content)  # type: ignore
    debug_print("ANALYSIS", analysis)
    return {'analysis': analysis}

# ── Critic Agent ──────────────────────────────────────────────
def critic_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un agent critique rigoureux. Évalue sur 10 :
    1. La qualité factuelle des recherches (0-10)
    2. La rigueur de l'analyse (0-10)
    3. Les informations manquantes ou contradictoires (0-10)
    Recherches : {state['research'][:_MAX_RESEARCH_CHARS]}
    Analyse : {state['analysis'][:_MAX_RESEARCH_CHARS]}

    Ta réponse DOIT se terminer par exactement cette ligne :
    SCORE: <entier de 0 à 10>

    Un score >= 7 est considéré satisfaisant. En dessous, fournis
    des instructions précises pour améliorer les recherches."""
    response = invoke_with_retry(prompt)
    critique = strip_think(response.content)  # type: ignore
    debug_print("CRITIC", critique)
    return {'critique': critique, 'iteration': state['iteration'] + 1}

# ── Synthesizer Agent ─────────────────────────────────────────
def synthesizer_node(state: AgentState) -> AgentState:
    prompt = f"""Synthétise une réponse finale claire et complète.
    Question : {state['query']}
    Recherches : {state['research'][:_MAX_RESEARCH_CHARS]}
    Analyse : {state['analysis'][:_MAX_RESEARCH_CHARS]}
    Formate la réponse avec des sections claires."""
    response = invoke_with_retry(prompt)
    final = strip_think(response.content)  # type: ignore
    debug_print("SYNTHESIS", final)
    return {'final_answer': final}

# ── Conditional Routing ──────────────────────────────────────
def should_retry(state: AgentState) -> str:
    critique = state["critique"]
    # Score extraction "SCORE: X"
    match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", critique)
    if match:
        score = float(match.group(1))
        debug_print("CRITIC SCORE", f"{score}/10")
    else:
        # Fallback : if the LLM did not follow the format, force a retry
        debug_print("CRITIC SCORE", "Non parseable — retry par défaut")
        score = 0.0

    # Threshold set to 7/10; safeguard on the number of iterations
    if score >= 7 or state["iteration"] >= 2:
        return "synthesize"
    return "retry"

# ── Graph Assembly ──────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node('orchestrator', orchestrator_node)
workflow.add_node('researcher', researcher_node)
workflow.add_node('analyst', analyst_node)
workflow.add_node('critic', critic_node)
workflow.add_node('synthesizer', synthesizer_node)

workflow.set_entry_point('orchestrator')
workflow.add_edge('orchestrator', 'researcher')
workflow.add_edge('researcher', 'analyst')
workflow.add_edge('analyst', 'critic')
workflow.add_conditional_edges(
    'critic',
    should_retry,
    {'synthesize': 'synthesizer', 'retry': 'researcher'}
)
workflow.add_edge('synthesizer', END)

app = workflow.compile()

# ── Exécution ─────────────────────────────────────────────────
REQ = "Quels sont les impacts économiques et sociaux de l'IA générative en France d'ici 2030 ?"
print(f"Query: {REQ}")
result = app.invoke({
    "query": REQ,
    "messages": []
})
print(result['final_answer'])

app.get_graph().draw_mermaid_png(output_file_path='graphScore.png')