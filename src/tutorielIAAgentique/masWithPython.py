# src/tutorielIAAgentique/masWithPython.py
import os
import json
import re
import operator
import subprocess
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
    code_results:        str
    analysis:            str
    critique:            str
    final_answer:        str
    iteration:           int
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
_THINK_RE      = re.compile(r'<think>.*?</think>', re.DOTALL)
_CODE_FENCE_RE = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)
_NUMERIC_HINTS = re.compile(
    r'\b(calculer?|compute|calcul|mean|average|écart.type|standard.deviation'
    r'|somme|sum|count|combien|how many|percentage|proportion'
    r'|median|mode|variance|regression|statistics|statistiques'
    r'|simulate|simuler|modèle numérique|numeric)\b',
    re.IGNORECASE,
)

def strip_think(text: str) -> str:
    return _THINK_RE.sub('', text).strip()

def search_web(query: str, max_results: int = 3) -> str:
    results = list(DDGS().text(query, max_results=max_results))
    return '\n'.join([f"- {r['title']}: {r['body']}" for r in results])

def execute_python(code: str, timeout: int = 5) -> str:
    """Exécute du code Python dans un sous-processus sandboxé."""
    try:
        result = subprocess.run(
            ['python', '-c', code],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        if errors:
            return f"STDERR: {errors}\nSTDOUT: {output}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "TIMEOUT: execution exceeded 5 seconds."
    except Exception as e:
        return f"ERROR: {e}"

def _extract_code(text: str) -> str | None:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else None

def self_reflect(answer: str, query: str) -> dict:
    """Auto-évaluation de la réponse sur 3 critères (1-5)."""
    prompt = (
        "Évalue ta propre réponse sur 3 critères (score entier de 1 à 5) :\n"
        "- completude : tous les aspects de la question sont-ils traités ?\n"
        "- precision   : les faits sont-ils corrects et sourcés ?\n"
        "- clarte      : la réponse est-elle bien structurée ?\n\n"
        f"Question : {query}\n"
        f"Réponse : {answer[:800]}\n\n"
        "Réponds avec UNIQUEMENT cet objet JSON (aucun texte avant ou après) :\n"
        '{"completude": X, "precision": X, "clarte": X, "commentaire": "..."}'
    )
    resp = llm.invoke(prompt)
    raw  = strip_think(resp.content)  # type: ignore

    # Extraction robuste du JSON
    json_match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group(0))
            if all(k in scores for k in ('completude', 'precision', 'clarte')):
                return scores
        except json.JSONDecodeError:
            pass

    # Fallback neutre — ne déclenche pas de RETRY intempestif
    return {
        'completude': 3, 'precision': 3, 'clarte': 3,
        'commentaire': f'parse error — raw: {raw[:200]!r}',
    }


# ══════════════════════════════════════════════════════════════
# Mini-ReAct coder — boucle interne du Researcher
# ══════════════════════════════════════════════════════════════
def mini_react_coder(task: str, context: str = '', max_steps: int = 3) -> str:
    system = (
        "You are a Python coding assistant. Write self-contained snippets "
        "that compute a result and print it to stdout.\n\n"
        "Rules:\n"
        "1. Always wrap code in ```python ... ``` fences.\n"
        "2. Use only the standard library (math, statistics, etc.).\n"
        "3. Always end with print().\n"
        "4. Fix errors if execution fails.\n"
        "5. When done, write: FINAL: <conclusion in plain text>."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": (
            f"Task: {task}\n"
            + (f"\nContext:\n{context[:600]}" if context else "")
        )},
    ]
    last_output = ""
    for step in range(max_steps):
        resp  = llm.invoke(messages)
        reply = strip_think(resp.content)  # type: ignore
        messages.append({"role": "assistant", "content": reply})

        if "FINAL:" in reply:
            final_line = [l for l in reply.splitlines() if "FINAL:" in l]
            return final_line[0].replace("FINAL:", "").strip() if final_line else reply

        code = _extract_code(reply)
        if not code:
            return reply.strip()

        debug_print(f"EXECUTE_PYTHON step {step+1}", code)
        output = execute_python(code)
        debug_print(f"EXECUTE_PYTHON output {step+1}", output)
        last_output = output
        messages.append({"role": "user", "content": f"Execution output:\n{output}"})

    return last_output or "No result produced."


# ══════════════════════════════════════════════════════════════
# Agents
# ══════════════════════════════════════════════════════════════
def orchestrator_node(state: AgentState) -> AgentState:
    prompt = (
        "Tu es un orchestrateur. Décompose cette question en tâches pour :\n"
        "  • un Researcher  (recherche de faits + calculs numériques)\n"
        "  • un Analyst     (analyse et raisonnement)\n"
        "Si la question implique des calculs, dis explicitement au Researcher "
        "d'utiliser execute_python.\n"
        "Sois concis.\n\n"
        f"Question : {state['query']}"
    )
    resp = llm.invoke(prompt)
    debug_print("ORCHESTRATOR PLAN", strip_think(resp.content))  # type: ignore
    return {
        'plan':                strip_think(resp.content),  # type: ignore
        'iteration':           0,
        'self_reflect_scores': {},
        'code_results':        '',
        'critique':            '',
        'final_answer':        '',
    }


def researcher_node(state: AgentState) -> AgentState:
    past_context = memory.retrieve(state['query'])[:600]
    debug_print("MEMORY CONTEXT", past_context)
    web_results  = search_web(state['query'])[:600]

    prompt = (
        "Tu es un agent de recherche avec mémoire.\n\n"
        f"Interactions passées pertinentes :\n{past_context}\n\n"
        f"Nouvelle question : {state['query']}\n"
        f"Plan : {state['plan'][:400]}\n\n"
        "Évite de répéter ce qui a déjà été traité. Complète et approfondis.\n"
        "Fournis des faits vérifiables et précise tes sources.\n\n"
        f"Résultats web récents :\n{web_results}"
    )
    resp     = llm.invoke(prompt)
    research = strip_think(resp.content)  # type: ignore
    debug_print("RESEARCH", research)

    code_results = ""
    if _NUMERIC_HINTS.search(state['plan']) or _NUMERIC_HINTS.search(state['query']):
        debug_print("RESEARCHER", "Numeric task detected — launching mini-ReAct coder")
        code_results = mini_react_coder(task=state['query'], context=web_results)
        debug_print("CODE RESULTS", code_results)
        research += f"\n\n## Computed Results (execute_python)\n{code_results}"

    return {'research': research, 'code_results': code_results}


def analyst_node(state: AgentState) -> AgentState:
    code_section = (
        f"\nRésultats calculés :\n{state['code_results'][:400]}"
        if state.get('code_results') else ""
    )
    prompt = (
        "Tu es un analyste expert. Sur la base des recherches, fournis une "
        "analyse approfondie et des conclusions.\n\n"
        f"Question : {state['query']}\n"
        f"Recherches : {state['research'][:700]}"
        f"{code_section}\n\n"
        "Identifie les limites et incertitudes."
    )
    resp = llm.invoke(prompt)
    analysis = strip_think(resp.content)  # type: ignore
    debug_print("ANALYSIS", analysis)
    return {'analysis': analysis}


def critic_node(state: AgentState) -> AgentState:
    prompt = (
        "Tu es un agent critique rigoureux. Évalue :\n"
        "1. La qualité factuelle des recherches\n"
        "2. La rigueur de l'analyse\n"
        "3. Les informations manquantes ou contradictoires\n\n"
        f"Recherches : {state['research'][:700]}\n"
        f"Analyse : {state['analysis'][:700]}\n\n"
        "Réponds par APPROVED si satisfaisant, ou RETRY suivi d'instructions "
        "précises si des améliorations sont nécessaires."
    )
    resp = llm.invoke(prompt)
    critique = strip_think(resp.content)  # type: ignore
    debug_print("CRITIC", critique)
    return {'critique': critique, 'iteration': state['iteration'] + 1}


def synthesizer_node(state: AgentState) -> AgentState:
    code_section = (
        f"\nRésultats calculés :\n{state['code_results'][:400]}"
        if state.get('code_results') else ""
    )
    prompt = (
        "Synthétise une réponse finale claire et complète.\n\n"
        f"Question : {state['query']}\n"
        f"Recherches : {state['research'][:500]}"
        f"{code_section}\n"
        f"Analyse : {state['analysis'][:500]}\n\n"
        "Formate la réponse avec des sections claires."
    )
    resp         = llm.invoke(prompt)
    final_answer = strip_think(resp.content)  # type: ignore
    debug_print("SYNTHESIS", final_answer)

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

    # RETRY uniquement si score vraiment bas ET quota non atteint
    if mean_score < 3.5 and state['iteration'] < 2:
        retry_msg = (
            f"RETRY – self-reflection score trop bas ({mean_score:.2f}/5). "
            f"{scores.get('commentaire', 'qualité insuffisante')}. "
            "Approfondir recherches et calculs."
        )
        debug_print("SELF-REFLECT DECISION", f"Score {mean_score:.2f}/5 < 3.5 → RETRY")
        return {
            'final_answer':        '',
            'critique':            retry_msg,
            'self_reflect_scores': scores,
            'iteration':           state['iteration'] + 1,
        }

    memory.store(
        query    = state['query'],
        answer   = final_answer,
        metadata = {"iteration": state['iteration'], "sr_mean": mean_score},
    )
    debug_print("SELF-REFLECT DECISION", f"Score {mean_score:.2f}/5 ≥ 3.5 → ACCEPTED")

    # ← critique réinitialisé pour stopper la boucle
    return {
        'final_answer':        final_answer,
        'self_reflect_scores': scores,
        'critique':            'APPROVED',
    }


# ══════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════
def should_retry(state: AgentState) -> str:
    if "APPROVED" in state["critique"] or state["iteration"] >= 2:
        return "synthesize"
    return "retry"

def after_synthesizer(state: AgentState) -> str:
    """Route vers END si réponse finale produite, sinon retry."""
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
    {'synthesize': 'synthesizer', 'retry': 'researcher'},
)
workflow.add_conditional_edges(
    'synthesizer',
    after_synthesizer,
    {'retry': 'researcher', END: END},
)

app = workflow.compile()


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════
requetes = [
    (
        "Calcule l'écart-type et la médiane des hauteurs (en mètres) "
        "des 10 plus hauts gratte-ciel de Paris : "
        "Tour Eiffel 330, Tour Montparnasse 210, Tour First 225, "
        "Tour Majunga 195, Tour CB21 183, Tour Eureka 165, "
        "Tour Initiale 163, Tour Opus 12 161, Tour Generali 155, Tour EDF 150."
    )
]

for req in requetes:
    print(f"\n{'=' * 70}\nQuery: {req}\n{'=' * 70}")
    result = app.invoke({
        "query":               req,
        "messages":            [],
        "self_reflect_scores": {},
        "code_results":        "",
    })
    print(result['final_answer'])
    scores = result.get('self_reflect_scores', {})
    if scores:
        mean = (
            scores.get('completude', 0)
            + scores.get('precision',  0)
            + scores.get('clarte',     0)
        ) / 3
        print(
            f"\n[Self-reflection] complétude={scores.get('completude')}  "
            f"précision={scores.get('precision')}  "
            f"clarté={scores.get('clarte')}  → moyenne={mean:.2f}/5"
        )

app.get_graph().draw_mermaid_png(output_file_path='graphWithPython.png')