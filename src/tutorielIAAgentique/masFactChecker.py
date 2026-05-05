# src/tutorielIAAgentique/masFactChecker.py
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
    """Retry call for  RateLimitError."""
    for attempt in range(retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            if '429' in str(e) and attempt < retries - 1:
                debug_print("RATE LIMIT", f"Pause {wait}s avant retry {attempt+1}/{retries}")
                time.sleep(wait)
            else:
                raise

llm = ChatGroq(model='qwen/qwen3-32b', temperature=0.0)



# ── State shared between all agents ───────────────────────────
class AgentState(TypedDict):
    query: str                   # Original user question
    plan: str                    # Orchestrator's decomposition plan
    research: str                # Researcher's raw findings
    fact_check: str              # FactChecker's verified claims report
    analysis: str                # Analyst's conclusions
    critique: str                # Critic's feedback
    final_answer: str            # Final synthesised answer
    iteration: int               # Retry counter (used by Critic routing)
    messages: Annotated[List, operator.add]  # Shared message history

# ── Helper: strip <think>…</think> reasoning blocks ───────────
# Qwen3 emits chain-of-thought tags that must not be treated as content.
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)

def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks and collapse extra blank lines."""
    return _THINK_RE.sub('', text).strip()


# ══════════════════════════════════════════════════════════════
# Orchestrator Agent
# ══════════════════════════════════════════════════════════════
def orchestrator_node(state: AgentState) -> AgentState:
    prompt = (
        "You are an orchestrator. Break the following question into tasks "
        "for two agents: a Researcher (fact gathering) and an Analyst "
        "(analysis and reasoning). Be concise.\n"
        f"Question: {state['query']}"
    )
    response = invoke_with_retry(prompt)
    plan = strip_think(response.content)  # type: ignore
    debug_print("ORCHESTRATOR PLAN", plan)
    return {'plan': plan, 'iteration': 0}


# ══════════════════════════════════════════════════════════════
# Researcher Agent
# ══════════════════════════════════════════════════════════════
def search_web(query: str, max_results: int = 3) -> str:
    """DuckDuckGo search — returns up to max_results snippets."""
    results = list(DDGS().text(query, max_results=max_results))
    return '\n'.join([f"- {r['title']}: {r['body']}" for r in results])

def researcher_node(state: AgentState) -> AgentState:
    web_results = search_web(state['query'])
    prompt = (
        "You are a research agent. Find factual information to answer:\n"
        f"{state['query']}\n\n"
        f"Plan: {state['plan']}\n\n"
        "Use your knowledge and be precise about sources. "
        "Provide verifiable facts.\n\n"
        f"Web results:\n{web_results}"
    )
    response = invoke_with_retry(prompt)
    research = strip_think(response.content)  # type: ignore
    debug_print("RESEARCH", research)
    return {'research': research}


# ══════════════════════════════════════════════════════════════
# FactChecker Agent  (inserted between Researcher and Analyst)
# ══════════════════════════════════════════════════════════════
# Strategy:
#   1. Ask the LLM to extract 3-5 key factual claims from the Researcher report.
#   2. Strip any <think> tags so only clean claim lines remain.
#   3. For each claim run an independent DuckDuckGo search (second call).
#   4. Ask the LLM to label each claim: CONFIRMED / UNCERTAIN / REFUTED.
#   5. Store the report in state['fact_check'] for the Analyst.
#
# Rate-limit guard: inputs are truncated and a short sleep separates LLM calls.

# Max characters fed to the LLM per verification call (keeps tokens < 1500).
_MAX_RESEARCH_CHARS = 800
_MAX_DDG_CHARS      = 400

def factchecker_node(state: AgentState) -> AgentState:
    # ── Step 1: extract key claims ────────────────────────────
    research_snippet = state['research'][:_MAX_RESEARCH_CHARS]
    extraction_prompt = (
        "From the research text below, list exactly 3 to 5 key factual "
        "claims (figures, dates, names, statistics). "
        "Return ONLY a numbered list, one claim per line, no extra commentary.\n\n"
        f"Research text:\n{research_snippet}"
    )
    claims_response = invoke_with_retry(extraction_prompt)
    raw_claims = strip_think(claims_response.content)  # type: ignore
    debug_print("FACTCHECKER — Extracted claims", raw_claims)

    # ── Step 2: filter and clean claim lines ──────────────────
    # Keep only lines that look like list items; skip blank lines and
    # any residual <think> content that slipped through.
    lines: list[str] = []
    for line in raw_claims.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are clearly chain-of-thought leftovers
        if line.startswith('<') or line.lower().startswith('okay') or line.lower().startswith('let'):
            continue
        # Strip leading list markers: "1.", "-", "•", ")", etc.
        claim = re.sub(r'^[\d]+[.)]\s*|^[-•*]\s*', '', line).strip()
        if len(claim) > 10:          # ignore very short / empty fragments
            lines.append(claim)

    if not lines:
        debug_print("FACTCHECKER — Report", "No verifiable claims could be extracted.")
        return {'fact_check': 'No verifiable claims could be extracted from the research.'}

    # ── Step 3 & 4: verify each claim via a second DDG search ─
    verified: list[str] = []
    for claim in lines:
        # Independent DuckDuckGo search for this specific claim
        ddg_results = search_web(claim, max_results=2)[:_MAX_DDG_CHARS]

        verdict_prompt = (
            "You are a fact-checking agent.\n"
            f'Claim to verify: "{claim}"\n\n'
            f"Web search results:\n{ddg_results}\n\n"
            "Reply with exactly one of the three labels below, "
            "followed by a one-sentence justification:\n"
            "- CONFIRMED  : the web sources corroborate the claim.\n"
            "- UNCERTAIN  : sources are insufficient or contradictory.\n"
            "- REFUTED    : sources clearly contradict the claim."
        )
        verdict_response = invoke_with_retry(verdict_prompt)
        verdict = strip_think(verdict_response.content).strip()  # type: ignore
        verified.append(f"• {claim}\n  → {verdict}")

        # Brief pause between calls to stay within TPM limits
        time.sleep(1)

    fact_check_report = "\n\n".join(verified)
    debug_print("FACTCHECKER — Report", fact_check_report)
    return {'fact_check': fact_check_report}


# ══════════════════════════════════════════════════════════════
# Analyst Agent
# ══════════════════════════════════════════════════════════════
# Receives both the raw research and the FactChecker report.
def analyst_node(state: AgentState) -> AgentState:
    fact_check = state.get('fact_check', 'Not available')
    # Truncate large inputs to avoid hitting the TPM limit
    research_snippet   = state['research'][:_MAX_RESEARCH_CHARS]
    fact_check_snippet = fact_check[:_MAX_RESEARCH_CHARS]
    prompt = (
        "You are an expert analyst. Based on the research and the "
        "fact-checking report, provide an in-depth analysis and conclusions.\n\n"
        f"Question: {state['query']}\n\n"
        f"Research (excerpt):\n{research_snippet}\n\n"
        f"Fact-checking report:\n{fact_check_snippet}\n\n"
        "Identify limits and uncertainties. Do not rely on REFUTED facts; "
        "clearly flag UNCERTAIN ones."
    )
    response = invoke_with_retry(prompt)
    analysis = strip_think(response.content)  # type: ignore
    debug_print("ANALYSIS", analysis)
    return {'analysis': analysis}


# ══════════════════════════════════════════════════════════════
# Critic Agent
# ══════════════════════════════════════════════════════════════
def critic_node(state: AgentState) -> AgentState:
    fact_check = state.get('fact_check', 'Not available')
    prompt = (
        "You are a rigorous critic agent. Evaluate:\n"
        "1. Factual quality of the research\n"
        "2. Rigour of the analysis\n"
        "3. Missing or contradictory information\n\n"
        f"Research: {state['research'][:600]}\n"
        f"Fact-checking report: {fact_check[:400]}\n"
        f"Analysis: {state['analysis'][:600]}\n\n"
        "Reply with APPROVED if satisfactory, or RETRY followed by "
        "specific improvement instructions if changes are needed."
    )
    response = invoke_with_retry(prompt)
    critique = strip_think(response.content)  # type: ignore
    debug_print("CRITIC", critique)
    return {'critique': critique, 'iteration': state['iteration'] + 1}


# ══════════════════════════════════════════════════════════════
# Synthesizer Agent
# ══════════════════════════════════════════════════════════════
def synthesizer_node(state: AgentState) -> AgentState:
    fact_check = state.get('fact_check', 'Not available')
    prompt = (
        "Synthesise a clear and complete final answer.\n\n"
        f"Question: {state['query']}\n\n"
        f"Research: {state['research'][:600]}\n"
        f"Fact-checking: {fact_check[:400]}\n"
        f"Analysis: {state['analysis'][:600]}\n\n"
        "Format the answer with clear sections."
    )
    response = invoke_with_retry(prompt)
    final = strip_think(response.content)  # type: ignore
    debug_print("SYNTHESIS", final)
    return {'final_answer': final}


# ══════════════════════════════════════════════════════════════
# Conditional routing
# ══════════════════════════════════════════════════════════════
def should_retry(state: AgentState) -> str:
    if "APPROVED" in state["critique"] or state["iteration"] >= 2:
        return "synthesize"
    return "retry"


# ══════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════
#
#   User Query
#       │
#       ▼
#  [Orchestrator]   ← plans and delegates
#       │
#       ▼
#   [Researcher]    ← fetches raw facts (DuckDuckGo #1)
#       │
#       ▼
#  [FactChecker]   ← extracts claims, verifies each via DuckDuckGo #2
#       │
#       ▼
#    [Analyst]      ← analyses verified facts
#       │
#       ▼
#    [Critic]       ← validates; may loop back to Researcher
#       │
#       ▼
#  [Synthesizer]   ← produces the final answer

workflow = StateGraph(AgentState)

workflow.add_node('orchestrator', orchestrator_node)
workflow.add_node('researcher',   researcher_node)
workflow.add_node('factchecker',  factchecker_node)
workflow.add_node('analyst',      analyst_node)
workflow.add_node('critic',       critic_node)
workflow.add_node('synthesizer',  synthesizer_node)

workflow.set_entry_point('orchestrator')
workflow.add_edge('orchestrator', 'researcher')
workflow.add_edge('researcher',   'factchecker')   # FactChecker sits here
workflow.add_edge('factchecker',  'analyst')
workflow.add_edge('analyst',      'critic')
workflow.add_conditional_edges(
    'critic',
    should_retry,
    {'synthesize': 'synthesizer', 'retry': 'researcher'}
)
workflow.add_edge('synthesizer', END)

app = workflow.compile()

# ── Run ────────────────────────────────────────────────────────
REQ = "Quels sont les impacts économiques et sociaux de l'IA générative en France d'ici 2030 ?"
print(f"Query: {REQ}")
result = app.invoke({
    "query": REQ,
    "messages": []
})
print(result['final_answer'])

app.get_graph().draw_mermaid_png(output_file_path='graphFactChecker.png')