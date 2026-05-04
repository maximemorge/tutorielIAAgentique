# tests/test_react_NewYork.py
from tutorielIAAgentique.tools import search_web, fetch_wikipedia, calculate
from tutorielIAAgentique.agent import react_agent

# ── Test: demonstrate a question to the agent
question = "What is the average height of the tallest 10 buildings in New York City?"
print(f"Question: {question}")
reponse = react_agent(question, max_steps=10)
print(f"Réponse: {reponse}")
