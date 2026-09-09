# llm-forge

Fine-tune a small language model in the cloud, then run *your* model locally.
A learning project that uses all three tools you have, each for what it's best at:

| Tool | Role | Does |
|------|------|------|
| **Claude Code** | engineering brain | builds & explains the repo (in VS Code) |
| **Google Colab** | GPU muscle | fine-tune, merge, convert to GGUF |
| **Ollama** | local runtime | serves your model, free & offline |

The one hop worth internalizing — the **GGUF bridge** — is what connects cloud
training to local inference:

```
Colab:  QLoRA fine-tune -> merge adapter -> convert to GGUF -> quantize
   |                                                              |
   +----------------------- download .gguf -----------------------+
                                                                  v
Local:  Modelfile (FROM ./model.gguf) -> ollama create -> ollama run
```

---

## The pipeline, step by step

Each file is tagged for **where it runs**. The rule of thumb: anything touching a
GPU or model weights runs in Colab; prep, packaging, serving, and eval run local.

| # | Step | Command | Runs |
|---|------|---------|------|
| 1 | Format data into chat JSONL | `make prep` | 🖥️ Local |
| 2 | QLoRA fine-tune | `make train` | ☁️ Colab |
| 3 | Merge adapter → fp16 | `make merge` | ☁️ Colab |
| 4 | Convert + quantize → GGUF | `make gguf` | ☁️ Colab |
| 5 | Download the `.gguf` | notebook cell 7 | ☁️ Colab |
| 6 | Import to Ollama & run | `make import` | 🖥️ Local |
| 7 | Compare base vs fine-tune | `make eval` | 🖥️ Local |

Steps 2–5 are orchestrated by `notebooks/colab_runner.ipynb` — open it in Colab,
set the GPU runtime, run top to bottom.

---

## Quickstart

**Prove it works end-to-end first**, on the bundled `sample_dataset.jsonl`, before
plugging in real data. That way any breakage is pipeline-shaped, not data-shaped.

1. **Local:** push this repo to GitHub from VS Code.
2. **Colab:** open `notebooks/colab_runner.ipynb`, set the repo URL, run all cells.
   You'll download `model.gguf` at the end.
3. **Local:** drop that file at `ollama/model.gguf`, then:
   ```bash
   make import          # ollama create + run — chat with your model
   make eval            # see what the fine-tune changed vs the base
   ```
4. Now swap in real data: implement `rows_from_raw()` in `data/prepare_data.py`,
   run `make prep`, flip `sample_path → train_path` in `config.yaml`, and rerun
   the Colab half.

---

## Repo layout

```
llm-forge/
├── config.yaml              # single source of truth (model, data, hyperparams)
├── Makefile                 # convenience targets, each tagged local/colab
├── requirements.txt         # 🖥️ local deps (tiny)
├── requirements-colab.txt   # ☁️ GPU training stack
├── data/
│   ├── prepare_data.py      # 🖥️ raw -> chat-template JSONL
│   └── sample_dataset.jsonl # runs the pipeline out of the box
├── train/
│   └── train_qlora.py       # ☁️ QLoRA fine-tune
├── export/
│   ├── merge_adapter.py     # ☁️ merge LoRA -> fp16
│   └── convert_to_gguf.py   # ☁️ the GGUF bridge (+ quantize)
├── ollama/
│   ├── Modelfile            # 🖥️ Ollama recipe (template, stops, params)
│   └── import.sh            # 🖥️ ollama create + run
├── eval/
│   └── compare.py           # 🖥️ base vs fine-tune, via local Ollama
└── notebooks/
    └── colab_runner.ipynb   # ☁️ orchestrates the GPU half
```

---

## Notes for Claude Code (hand-off)

- **Start in `config.yaml`.** Every script reads it; change values there, not in code.
- **Version drift is expected.** The TRL / transformers / peft APIs move quickly.
  If an import or argument fails, reconcile against installed versions rather than
  assuming the script is wrong — the structure is stable, the exact signatures drift.
  The likely spots: `SFTConfig`/`SFTTrainer` args in `train/train_qlora.py`
  (e.g. `processing_class` vs `tokenizer`, `max_length` vs `max_seq_length`).
- **The Modelfile template is model-family-specific.** It's set for Qwen2.5
  (`<|im_start|>`). If you change `base_id`, swap the template + stop tokens to match
  (`ollama show --modelfile <base>` is a good source).
- **`target_modules` in the LoRA config** are named for Qwen/Llama-style architectures.
  A different family may use different projection names.
- Good next rungs once this works: an LLM-as-judge eval harness (extend `eval/`),
  then a fully-offline RAG lab reusing the same Ollama runtime.

---

## What you'll actually learn

QLoRA and quantization • chat templates & special tokens • the adapter→merge→GGUF
artifact chain • local model serving • and the first taste of evaluation — the
skill that matters most in production.
