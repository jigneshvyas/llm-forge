"""
[RUNS: LOCAL or COLAB — CPU only]

Turn whatever raw source you have into the chat-template JSONL the trainer expects.

Output format (one JSON object per line):
    {"messages": [
        {"role": "system",    "content": "..."},   # optional
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}

This "messages" (conversational) format is what modern TRL SFTTrainer consumes
directly — it applies the base model's chat template for you, so you never
hand-write special tokens like <|im_start|>. That's the whole point of this file:
get your data into this shape, and the rest of the pipeline just works.

TODO (hand to Claude Code):
  - Point RAW_INPUT at your source (CSV of tickets, JSONL of Q&A, scraped docs...).
  - Implement `rows_from_raw()` for your specific schema.
  - Decide your system prompt — it defines the persona/behavior you're training in.
"""

import json
from pathlib import Path

RAW_INPUT = "data/raw/your_source_here.csv"   # TODO: your real source
TRAIN_OUT = "data/train.jsonl"
EVAL_OUT = "data/eval.jsonl"
EVAL_FRACTION = 0.1

SYSTEM_PROMPT = "You are a concise, accurate assistant."  # TODO: your domain persona


def rows_from_raw():
    """
    Yield (user_text, assistant_text) pairs from your raw source.

    TODO: replace this stub. Example for a CSV with 'question','answer' columns:

        import csv
        with open(RAW_INPUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield row["question"], row["answer"]
    """
    raise NotImplementedError("Implement rows_from_raw() for your data schema.")


def to_record(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def main():
    records = [to_record(u, a) for u, a in rows_from_raw()]
    if not records:
        raise SystemExit("No records produced — check rows_from_raw().")

    split = max(1, int(len(records) * EVAL_FRACTION))
    eval_records, train_records = records[:split], records[split:]

    for path, recs in [(TRAIN_OUT, train_records), (EVAL_OUT, eval_records)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(recs):>5} records -> {path}")


if __name__ == "__main__":
    main()
