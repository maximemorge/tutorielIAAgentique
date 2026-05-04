# tests/test_react_Tokyo.py
from tutorielIAAgentique.tools import search_web, fetch_wikipedia, calculate
from tutorielIAAgentique.agent import react_agent
question = (
    "What are the top 3 most visited cities in the world according to recent rankings, and what is the current temperature in each of them?"
)
print(f"Question: {question}\n")
answer = react_agent(question)
print(f"\nFinal Answer: {answer}")
