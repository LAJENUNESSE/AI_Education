"""测试 Ollama 代码审查."""
import json, subprocess, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from app import REVIEW_PROMPT, OLLAMA_EXE, OLLAMA_MODEL

print(f"Model: {OLLAMA_MODEL}")

code = '''def greet(name):
    return 'Hello ' + name

print(greet("World"))'''

prompt = REVIEW_PROMPT.format(code=code)
print("=== PROMPT ===")
print(prompt)
print("\n=== CALLING OLLAMA ===")

result = subprocess.run(
    [OLLAMA_EXE, 'run', OLLAMA_MODEL, prompt],
    capture_output=True, text=True, timeout=120,
    env={**os.environ, 'OLLAMA_HOST': '127.0.0.1:11434'}
)

raw = result.stdout.strip()
raw = re.sub(r'\x1b\[[0-9?;]*[a-zA-Z]', '', raw)
raw = re.sub(r'\.\.\.done thinking\.', '', raw).strip()

print("=== RAW OUTPUT ===")
print(raw)
