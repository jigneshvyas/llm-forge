"""
[RUNS: COLAB — GPU REQUIRED (T4 is enough for a 1–2B model)]

QLoRA fine-tune: freeze the base model in 4-bit, train only LoRA adapters.
Reads all hyperparameters from config.yaml. Writes the adapter to train.output_dir.

The flow:
  1. Load base model in 4-bit (bitsandbytes) — this is the "Q" in QLoRA.
  2. Attach LoRA adapters to the target modules — the only trainable weights.
  3. SFTTrainer consumes the conversational JSONL and applies the chat template.
  4. Save the adapter (a few MB), NOT the whole model.

NOTE ON VERSIONS: the TRL / transformers / peft APIs move fast. If an import or
argument name fails, that's expected — ask Claude Code to reconcile against the
installed versions (`pip show trl transformers peft`). The structure below is
the stable part; the exact signatures are the part that drifts.
"""

import inspect

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    m, l, t, d = cfg["model"], cfg["lora"], cfg["train"], cfg["data"]

    # --- 1. 4-bit quantization config (QLoRA's memory trick) ------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # normal-float 4-bit, the QLoRA default
        bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16, store in 4-bit
        bnb_4bit_use_double_quant=True,         # extra memory saving
    )

    tokenizer = AutoTokenizer.from_pretrained(m["base_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # common fix for models lacking a pad token

    model = AutoModelForCausalLM.from_pretrained(
        m["base_id"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)  # enables grad checkpointing etc.

    # --- 2. LoRA adapters — the only trainable parameters ---------------------
    lora_config = LoraConfig(
        r=l["r"],
        lora_alpha=l["alpha"],
        lora_dropout=l["dropout"],
        target_modules=l["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 3. Data. TRL applies the chat template to the "messages" field. ------
    # Swap sample_path -> train_path once you've proven the pipeline works.
    dataset = load_dataset("json", data_files=d["sample_path"], split="train")

    # TRL's SFTConfig accepted kwargs drift across versions (e.g. warmup_ratio /
    # max_length have moved or been renamed before). Filter against whatever the
    # installed version actually accepts instead of failing on a TypeError.
    sft_kwargs = {
        "output_dir": t["output_dir"],
        "num_train_epochs": t["epochs"],
        "per_device_train_batch_size": t["batch_size"],
        "gradient_accumulation_steps": t["grad_accum"],
        "learning_rate": t["lr"],
        "max_length": t["max_seq_len"],
        "max_seq_length": t["max_seq_len"],  # older trl name for the same setting
        "warmup_ratio": t["warmup_ratio"],
        "logging_steps": t["logging_steps"],
        "save_steps": t["save_steps"],
        "seed": t["seed"],
        "bf16": True,
        "report_to": "none",
    }
    accepted = set(inspect.signature(SFTConfig.__init__).parameters)
    dropped = {k: v for k, v in sft_kwargs.items() if k not in accepted}
    sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in accepted}
    if dropped:
        print(f"NOTE: installed trl's SFTConfig doesn't accept {sorted(dropped)}; "
              f"dropping (check `pip show trl` if this matters for your run).")

    sft = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,  # newer TRL name; older versions use tokenizer=
    )

    # Watch train loss fall — that's the model learning your data.
    trainer.train()

    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"\nAdapter saved to {t['output_dir']}")
    print("Next: export/merge_adapter.py -> export/convert_to_gguf.py")


if __name__ == "__main__":
    main()
