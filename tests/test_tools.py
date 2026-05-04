# tests/test_tools.py
from tutorielIAAgentique.tools import search_web, fetch_wikipedia, calculate

# ── Test 1 : search_web ──────────────────────────────────
print("=" * 50)
print("TEST 1 : search_web — population du Japon")
print("=" * 50)
result1 = search_web("population du Japon")
print(result1)

# ── Test 2 : fetch_wikipedia ─────────────────────────────
print("=" * 50)
print("TEST 2 : fetch_wikipedia — Alan Turing")
print("=" * 50)
result2 = fetch_wikipedia("Alan Turing")
print(result2)

# ── Test 3 : calculate ───────────────────────────────────
print("=" * 50)
print("TEST 3 : calculate — 294 / 7")
print("=" * 50)
result3 = calculate("294 / 7")
print(result3)