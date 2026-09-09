"""
[RUNS: LOCAL — via Ollama's HTTP API at localhost:11434]

Step 7 of the project, and your first taste of evaluation (Project 3 on the ladder).

Run the SAME prompts through the base model and your fine-tune, side by side, so
you can SEE what the fine-tune changed. No GPU, no API cost — both models run
locally through Ollama.

Prereqs:
  - Your fine-tune imported:  ollama create forge-demo -f ollama/Modelfile
  - The base pulled for comparison:  ollama pull qwen2.5:1.5b-instruct
    (match whatever base you actually fine-tuned)
"""

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

BASE_MODEL = "qwen2.5:1.5b-instruct"   # TODO: match your config.yaml base
TUNED_MODEL = "forge-demo"             # TODO: match config.yaml ollama.model_name

PROMPTS = [
    "In one sentence, what is QLoRA?",
    "Why quantize a model before running it locally?",
    "What role does a Modelfile play in Ollama?",
    # TODO: add prompts that probe the behavior you actually trained in.
]


def chat(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def main():
    for prompt in PROMPTS:
        print("=" * 70)
        print(f"PROMPT: {prompt}\n")
        print(f"[BASE  {BASE_MODEL}]\n{chat(BASE_MODEL, prompt)}\n")
        print(f"[TUNED {TUNED_MODEL}]\n{chat(TUNED_MODEL, prompt)}\n")

    # Next rung (Project 3): instead of eyeballing, have a local judge model
    # score each pair on correctness/style and tally the results.


if __name__ == "__main__":
    main()
