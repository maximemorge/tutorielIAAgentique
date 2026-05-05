# src/agents/masWithMemory.py
#
# Multi-agent system with persistent vector memory.
# Agents communicate via a shared blackboard (AgentState).
# Memory is stored in a FAISS vector store and retrieved by the Researcher
# at the start of each run, allowing the system to build on past answers.
# src/tutorielIAAgentique/masFactChecker.py
import os
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

# ── Local embeddings (free, no API key required) ──────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ══════════════════════════════════════════════════════════════
# Shared state (blackboard architecture)
# All agents read from and write to this structure.
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    query: str                              # Original user question
    plan: str                               # Orchestrator's task decomposition
    research: str                           # Researcher's raw findings
    analysis: str                           # Analyst's conclusions
    critique: str                           # Critic's feedback
    final_answer: str                       # Final synthesised answer
    iteration: int                          # Retry counter
    messages: Annotated[List, operator.add] # Shared message history


# ══════════════════════════════════════════════════════════════
# Persistent vector memory
# Stores Q/A pairs as FAISS documents and retrieves the k most
# semantically similar past interactions for a given query.
# ══════════════════════════════════════════════════════════════
class AgentMemory:
    """Persistent vector memory for agents (FAISS + HuggingFace embeddings)."""

    def __init__(self):
        self.vectorstore = None          # Lazily created on first store() call
        self.conversation_history = []   # Optional: keep a flat log

    def store(self, query: str, answer: str, metadata: dict = {}) -> None:
        """Embed and store a Q/A pair in the vector store."""
        doc = Document(
            page_content=f"Q: {query}\nA: {answer}",
            metadata={'query': query, **metadata}
        )
        if self.vectorstore is None:
            # First document: initialise the FAISS index
            self.vectorstore = FAISS.from_documents([doc], embeddings)
        else:
            self.vectorstore.add_documents([doc])

    def retrieve(self, query: str, k: int = 3) -> str:
        """Return the k most relevant past interactions as a formatted string."""
        if self.vectorstore is None:
            return "Aucune mémoire disponible."
        docs = self.vectorstore.similarity_search(query, k=k)
        return '\n\n'.join([d.page_content for d in docs])

# Single shared memory instance used by all agents in this session
memory = AgentMemory()

# ══════════════════════════════════════════════════════════════
# Helper: web search
# ══════════════════════════════════════════════════════════════
def search_web(query: str) -> str:
    """DuckDuckGo search — returns the top 3 result snippets."""
    results = list(DDGS().text(query, max_results=3))
    return '\n'.join([f"- {r['title']}: {r['body']}" for r in results])

# ══════════════════════════════════════════════════════════════
# Orchestrator Agent
# Plans and delegates to Researcher and Analyst.
# ══════════════════════════════════════════════════════════════
def orchestrator_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un orchestrateur. Décompose cette question en tâches
    pour deux agents : un Researcher (recherche de faits) et un Analyst
    (analyse et raisonnement). Sois concis.
    Question : {state["query"]}"""
    response = llm.invoke(prompt)
    debug_print("ORCHESTRATOR PLAN", response.content)  # type: ignore
    return {'plan': response.content, 'iteration': 0}

# ══════════════════════════════════════════════════════════════
# Researcher Agent — with memory
# Retrieves relevant past interactions before searching the web,
# so it can deepen rather than repeat previous answers.
# ══════════════════════════════════════════════════════════════
def researcher_node(state: AgentState) -> AgentState:
    past_context = memory.retrieve(state["query"])
    # ── CORRECTIF : tronquer pour rester dans les limites TPM ──
    past_context = past_context[:600]          # ← ajouté
    debug_print("MEMORY CONTEXT", past_context)
    web_results = search_web(state['query'])
    web_results = web_results[:400]            # ← ajouté
    prompt = f"""Tu es un agent de recherche avec mémoire.

    Interactions passées pertinentes (résumé) :
    {past_context}

    Nouvelle question : {state['query']}
    Plan : {state['plan'][:300]}

    Évite de répéter ce que tu as déjà traité. Complète et approfondis.
    Fournis des faits vérifiables et précise tes sources.

    Résultats web récents :
    {web_results}"""
    response = llm.invoke(prompt)
    debug_print("RESEARCH", response.content)
    return {'research': response.content}

# ══════════════════════════════════════════════════════════════
# Analyst Agent
# Analyses the Researcher's findings and draws conclusions.
# ══════════════════════════════════════════════════════════════
def analyst_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un analyste expert. Sur la base des recherches,
    fournis une analyse approfondie et des conclusions.
    Question : {state['query']}
    Recherches : {state['research'][:800]}
    Identifie les limites et incertitudes."""
    response = llm.invoke(prompt)
    debug_print("ANALYSIS", response.content)
    return {'analysis': response.content}

# ══════════════════════════════════════════════════════════════
# Critic Agent
# Validates quality; triggers a retry if the answer is insufficient.
# Hard cap at 2 iterations to avoid infinite loops.
# ══════════════════════════════════════════════════════════════
def critic_node(state: AgentState) -> AgentState:
    prompt = f"""Tu es un agent critique rigoureux. Évalue :
    1. La qualité factuelle des recherches
    2. La rigueur de l'analyse
    3. Les informations manquantes ou contradictoires
    Recherches : {state['research'][:800]}
    Analyse : {state['analysis'][:800]}
    Réponds par APPROVED si satisfaisant, RETRY suivi d'instructions
    si des améliorations sont nécessaires."""
    response = llm.invoke(prompt)
    debug_print("CRITIC", response.content)
    return {'critique': response.content, 'iteration': state['iteration'] + 1}

# ══════════════════════════════════════════════════════════════
# Synthesizer Agent
# Produces the final, well-structured answer for the user.
# Stores the Q/A pair in memory so future runs can build on it.
# Reserch and analysis are truncated
# ══════════════════════════════════════════════════════════════
def synthesizer_node(state: AgentState) -> AgentState:
    prompt = f"""Synthétise une réponse finale claire et complète.
    Question : {state['query']}
    Recherches : {state['research'][:500]}
    Analyse : {state['analysis'][:500]}
    Formate la réponse avec des sections claires."""
    response = llm.invoke(prompt)
    debug_print("SYNTHESIS", response.content)  # type: ignore
    final_answer = response.content
    # Persist this Q/A pair so future runs can build on it
    memory.store(
        query=state['query'],
        answer=final_answer,
        metadata={"iteration": state['iteration']}
    )
    debug_print("MEMORY STORED", f"Stored answer for: {state['query'][:80]}...")
    return {'final_answer': final_answer}


# ══════════════════════════════════════════════════════════════
# Conditional routing
# Routes back to Researcher (retry) or forward to Synthesizer.
# ══════════════════════════════════════════════════════════════
def should_retry(state: AgentState) -> str:
    if "APPROVED" in state["critique"] or state["iteration"] >= 2:
        return "synthesize"
    return "retry"


# ══════════════════════════════════════════════════════════════
# Graph assembly
#
#   User Query
#       │
#       ▼
#  [Orchestrator] ← decomposes the task
#       │
#       ▼
#  [Researcher] ← fetches facts + queries memory
#       │
#       ▼
#   [Analyst] ← analyses findings
#       │
#       ▼
#   [Critic] ← validates; may loop back to Researcher
#       │
#       ▼
#  [Synthesizer] ← produces final answer + stores in memory
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
workflow.add_edge('synthesizer', END)

app = workflow.compile()


# ══════════════════════════════════════════════════════════════
# Run — 3 successive queries on the same theme.
# Memory accumulates across queries: by the 3rd query the
# Researcher receives 2 past Q/A pairs and deepens its answer.
#
# Expected debug output:
#   Query 1 → MEMORY CONTEXT: "Aucune mémoire disponible."
#   Query 2 → MEMORY CONTEXT: [answer from query 1]
#   Query 3 → MEMORY CONTEXT: [answers from queries 1 and 2]
# ══════════════════════════════════════════════════════════════
requetes = [
    "Quels sont les impacts économiques de l'IA générative en France d'ici 2030 ?",
    "Quels secteurs français seront les plus touchés par l'IA générative ?",
    "Comment la France se prépare-t-elle aux transformations sociales liées à l'IA ?"
]

for req in requetes:
    print(f"\n{'=' * 60}\nQuery: {req}\n{'=' * 60}")
    result = app.invoke({
        "query": req,
        "messages": []
    })
    # Note: memory.store() is called inside synthesizer_node,
    # no additional store() call needed here.
    print(result['final_answer'])

# Export the workflow graph as a PNG for visualisation
app.get_graph().draw_mermaid_png(output_file_path='graphWithMemory.png')