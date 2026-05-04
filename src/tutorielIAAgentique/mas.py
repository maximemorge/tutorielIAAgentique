# src/tutorielIAAgentique/mas.py
import os
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from ddgs import DDGS
from groq import Groq
from dotenv import load_dotenv
from tutorielIAAgentique.utils import debug_print
from tutorielIAAgentique.tools import TOOLS
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / '.env')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
llm = ChatGroq(model='qwen/qwen3-32b', temperature=0.0)

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
    response = llm.invoke(prompt)
    debug_print("ORCHESTRATOR PLAN", response.content) # type: ignore
    return {'plan': response.content, 'iteration': 0}


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
    response = llm.invoke(prompt_with_context)
    debug_print("RESEARCH", response.content) # type: ignore
    return {'research': response.content}

# ── Analyst Agent ─────────────────────────────────────────────
def analyst_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un analyste expert. Sur la base des recherches,
    fournis une analyse approfondie et des conclusions.
    Question : {state['query']}
    Recherches : {state['research']}
    Identifie les limites et incertitudes."""
    response = llm.invoke(prompt)
    debug_print("ANALYSIS", response.content) # type: ignore
    return {'analysis': response.content}

# ── Critic Agent ──────────────────────────────────────────────
def critic_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un agent critique rigoureux. Évalue :
    1. La qualité factuelle des recherches
    2. La rigueur de l'analyse
    3. Les informations manquantes ou contradictoires
    Recherches : {state['research']}
    Analyse : {state['analysis']}
    Réponds par APPROVED si satisfaisant, RETRY suivi d'instructions
    si des améliorations sont nécessaires."""
    response = llm.invoke(prompt)
    debug_print("CRITIC", response.content) # type: ignore
    return {'critique': response.content, 'iteration': state['iteration']+1}

# ── Synthesizer Agent ─────────────────────────────────────────
def synthesizer_node(state: AgentState) -> AgentState:
    prompt = f"""Synthétise une réponse finale claire et complète.
    Question : {state['query']}
    Recherches : {state['research']}
    Analyse : {state['analysis']}
    Formate la réponse avec des sections claires."""
    response = llm.invoke(prompt)
    debug_print("SYNTHESIS", response.content) # type: ignore
    return {'final_answer': response.content}

# ── Conditional Routing ──────────────────────────────────────
def should_retry(state: AgentState) -> str:
    if "APPROVED" in state["critique"] or state["iteration"] >= 2:
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

app.get_graph().draw_mermaid_png(output_file_path='graph.png')