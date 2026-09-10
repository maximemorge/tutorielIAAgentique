# src/tutorielIAAgentique/agent.py
import os
import re
import time
from groq import Groq
from groq import APIConnectionError, APIStatusError, RateLimitError
from dotenv import load_dotenv
from tutorielIAAgentique.utils import debug_print
from tutorielIAAgentique.tools import TOOLS
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / '.env')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

debug = True

MAX_RETRIES = 4

def _invoke_llm(messages: list) -> str:
    """Appel LLM avec retry sur erreurs transitoires (rate limit / réseau)."""
    delay = 3
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=messages,
                temperature=0,
                seed=42,
                max_tokens=800,
                stop=["---END---"]
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            code = getattr(getattr(e, 'response', None), 'status_code', None)
            transient = isinstance(e, APIConnectionError) or isinstance(e, RateLimitError) \
                or (code is not None and code >= 500)
            if transient and attempt < MAX_RETRIES:
                attempt += 1
                if debug:
                    debug_print("RETRY", f"Erreur transitoire ({type(e).__name__}), nouvel essai {attempt}/{MAX_RETRIES} dans {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise

# Read prompt
with open(os.path.join(os.path.dirname(__file__), 'prompt.md'), encoding='utf-8') as prompt_file:
    SYSTEM_PROMPT = prompt_file.read()

# Extract thought from the LLM output
def parse_thought(text: str):
    """Extracts thought from the LLM output."""
    match = re.search(r"^Thought:\s*(.+)", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

# Extract tool call from the LLM output
def parse_action(text: str):
    """Extracts tools from the LLM output."""
    match = re.search(r'Action:\s*(\w+)\(\s*["\']?(.+?)["\']?\s*\)', text, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip().strip('"\'')
    return None, None

# ReAct loop
def react_agent(question: str, max_steps: int = 6) -> str:
    """Implementation of the ReAct loop."""
    # 1. Initial call to the LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    # 2. Iterate (with max steps)
    for step in range(max_steps):
        # 2.a. Get the answer
        reply = _invoke_llm(messages)
        if debug :
            debug_print("RESPONSE", reply)
        # 2.b. Add it to the message list
        messages.append({'role': 'assistant', 'content': reply})
        # 2.c. If the answer is final, return the result
        if 'Final Answer:' in reply:
            return reply.split('Final Answer:')[-1].strip()
        # 2.d. Otherwise, call the tool and add the result to the messages list
        reply_clean = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
        # filter out the tool call from the assistant's reply'
        tool_name, tool_input = parse_action(reply)
        if tool_name and tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_input)
            obs_msg = f'Observation: {observation[:800]}'
            if debug:
                debug_print("OBSERVATION", obs_msg)
                debug_print(f"STEP {step + 1}", f"{tool_name}({tool_input[:50]}...)")
            messages.append({'role': 'user', 'content': obs_msg})
        else:
            if debug:
                debug_print("WARNING", f"Could not parse action from reply:\n{reply}")
            break
    # in case the number of steps is not enough
    return 'Maximum number of steps reached.'

