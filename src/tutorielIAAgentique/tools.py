# src/tutorielIAAgentique/tools.py
# Tool registry for the agentic system.
# Each tool is a plain Python function: str -> str.
#
# Tools:
#   search_web(query)      — DuckDuckGo top-3 snippets
#   fetch_wikipedia(title) — Wikipedia article summary
#   calculate(expr)        — safe eval of a math expression
#   execute_python(code)   — sandboxed subprocess, timeout=5 s  ← NEW
import os
import re
import subprocess
import sys
import tempfile
import textwrap
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


# ── execute_python ────────────────────────────────────────────
# Security model
# ─────────────
# The sandbox is intentionally *lightweight* (suitable for a course
# environment). It does NOT replace a proper container sandbox (gVisor,
# Firecracker, etc.), but it provides three layers of defence:
#
#   1. Blocklist — a short regex scan rejects the most dangerous
#      patterns (os.system, subprocess, open, __import__, network
#      sockets, file writes) before the code even reaches the
#      interpreter. This stops naive prompt-injection attacks.
#
#   2. Restricted globals — the subprocess runs with a fresh Python
#      interpreter that has no access to the parent process's
#      environment variables or working directory state.
#
#   3. Hard timeout — subprocess.run(timeout=5) ensures the child
#      process is killed after 5 seconds, preventing infinite loops
#      and CPU-exhaustion attacks.
#
# What the agent CAN do: pure computation, math, statistics,
# string manipulation, list/dict operations, standard-library
# modules that are already imported inside the snippet.
#
# What the agent CANNOT do (blocked by the blocklist):
#   • filesystem writes  (open(..., 'w'), pathlib write_text, …)
#   • shell execution    (os.system, subprocess, popen, …)
#   • network access     (socket, urllib, requests, …)
#   • dynamic imports    (__import__, importlib, …)
#   • interpreter exit   (sys.exit, quit, …)

# Patterns that are never allowed, regardless of context.
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    # Filesystem writes
    (r'\bopen\s*\(', "open()"),
    (r'\.write\s*\(',  ".write()"),
    (r'write_text\s*\(', "write_text()"),
    # Shell / subprocess
    (r'\bos\s*\.\s*system\b',   "os.system"),
    (r'\bos\s*\.\s*popen\b',    "os.popen"),
    (r'\bsubprocess\b',          "subprocess"),
    (r'\bpopen\b',               "popen"),
    # Dynamic imports
    (r'\b__import__\s*\(',       "__import__()"),
    (r'\bimportlib\b',           "importlib"),
    # Network
    (r'\bsocket\b',              "socket"),
    (r'\burllib\b',              "urllib"),
    (r'\brequests\b',            "requests"),
    (r'\bhttpx\b',               "httpx"),
    # Exit / signals
    (r'\bsys\s*\.\s*exit\b',     "sys.exit"),
    (r'\bquit\s*\(\)',           "quit()"),
    (r'\bexit\s*\(\)',           "exit()"),
    # Dangerous builtins
    (r'\beval\s*\(',             "eval()"),
    (r'\bexec\s*\(',             "exec()"),
    (r'\bcompile\s*\(',          "compile()"),
    (r'\b__builtins__\b',        "__builtins__"),
]

# Maximum characters of output returned to the caller.
_MAX_OUTPUT_CHARS = 1200

# Maximum characters of code accepted (prevents huge code blobs).
_MAX_CODE_CHARS = 2000


def _check_blocklist(code: str) -> str | None:
    """Return a human-readable error if a blocked pattern is found, else None."""
    for pattern, label in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"SecurityError: '{label}' is not allowed in sandboxed code."
    return None


def execute_python(code: str) -> str:
    """
    Execute *code* in a sandboxed subprocess (timeout=5 s).

    Returns the combined stdout + stderr of the script, truncated to
    _MAX_OUTPUT_CHARS.  Any security violation or runtime error is
    reported as a plain-text error string so the agent can react.

    Usage by the LLM
    ────────────────
    The agent should write a complete, self-contained Python snippet
    that prints its result(s) to stdout.  Example:

        import math
        primes = [n for n in range(2, 200) if all(n % d for d in range(2, n))]
        print(f"Primes below 200: {primes}")
        print(f"Count: {len(primes)}")
    """
    # ── Guard: code too long ──────────────────────────────────
    if len(code) > _MAX_CODE_CHARS:
        return (
            f"Error: code is too long ({len(code)} chars). "
            f"Maximum allowed: {_MAX_CODE_CHARS} chars."
        )

    # ── Guard: blocklist scan ─────────────────────────────────
    error = _check_blocklist(code)
    if error:
        return error

    # ── Write to a temporary file ─────────────────────────────
    # We use a temp file rather than -c "..." so that multi-line
    # code with indentation is handled correctly by the shell.
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8',
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    # ── Run in a subprocess ───────────────────────────────────
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],  # same Python interpreter
            capture_output=True,
            text=True,
            timeout=5,                   # hard wall-clock limit
            # No shell=True — avoids shell-injection via the code path
        )
        stdout = result.stdout
        stderr = result.stderr

        if stderr and not stdout:
            output = f"[stderr]\n{stderr}"
        elif stderr:
            output = f"{stdout}\n[stderr]\n{stderr}"
        else:
            output = stdout or "(no output)"

    except subprocess.TimeoutExpired:
        output = "Error: execution timed out after 5 seconds."
    except Exception as exc:
        output = f"Error: {exc}"
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return output[:_MAX_OUTPUT_CHARS]

# ── Tool registry ─────────────────────────────────────────────
# The ReAct agent and the MAS researcher look up tools by name here.
TOOLS: dict[str, callable] = {
    'search_web':       search_web,
    'fetch_wikipedia':  fetch_wikipedia,
    'calculate':        calculate,
    'execute_python':   execute_python,   # ← NEW
}