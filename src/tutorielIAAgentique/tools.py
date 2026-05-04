# src/tutorielIAAgentique/tools.py
# Tool registry for the agentic system.
# Each tool is a plain Python function: str -> str.
#
# Tools:
#   search_web(query)      — DuckDuckGo top-3 snippets
#   fetch_wikipedia(title) — Wikipedia article summary
#   calculate(expr)        — safe eval of a math expression
from ddgs import DDGS
from wikipediaapi import Wikipedia

# ── search_web ────────────────────────────────────────────────
def search_web(query: str) -> str:
    """DuckDuckGo search, returns first 3 results."""
    results = list(DDGS().text(query, max_results=3))
    return '\n'.join([f"- {r['title']}: {r['body']}" for r in results])

# ── fetch_wikipedia ───────────────────────────────────────────
def fetch_wikipedia(title: str) -> str:
    """Wikipedia search, returns summary of the article (≤ 1500 chars)."""
    wiki = Wikipedia(user_agent='AgenticAI/0.0 https://afia.asso.fr', language='en')
    page = wiki.page(title)
    return page.summary[:1500] if page.exists() else 'Article not found.'

# ── calculate ─────────────────────────────────────────────────
def calculate(expr: str) -> str:
    """Safely evaluate a Python math expression (no builtins)."""
    try:
        return str(eval(expr, {'__builtins__': {}}, {}))
    except Exception as e:
        return f'Error: {e}'

# ── Tool registry ─────────────────────────────────────────────
TOOLS: dict[str, callable] = {
    'search_web':       search_web,
    'fetch_wikipedia':  fetch_wikipedia,
    'calculate':        calculate
}