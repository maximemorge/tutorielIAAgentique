# tests/test_react_Tokyo.py
from tutorielIAAgentique.tools import search_web, fetch_wikipedia, calculate
from tutorielIAAgentique.agent import react_agent

# ── Test: demonstrate a question to the agent
question = "What is the population of Tokyo in 2024 and how many times bigger is it compared to the population of Paris?"
print(f"Question: {question}")
reponse = react_agent(question)
print(f"Réponse: {reponse}")
