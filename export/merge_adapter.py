"""
[RUNS: COLAB — GPU helpful, CPU works if RAM allows]

Merge the LoRA adapter back into the base model and save a standalone fp16 model.

Why this step exists: the adapter alone can't be converted to GGUF. GGUF wants a
complete model. So we load the base in full precision (NOT 4-bit — we want clean
weights to fold the adapter into), apply merge_and_unload(), and write out a
normal HuggingFace model directory that convert_to_gguf.py can consume.
"""

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    base_id = cfg["model"]["base_id"]
    adapter_dir = cfg["train"]["output_dir"]
    merged_dir = cfg["model"]["merged_dir"]

    print(f"Loading base {base_id} in fp16...")
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Applying adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()  # folds LoRA deltas into the base tensors

    print(f"Saving merged fp16 model to {merged_dir}...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(merged_dir)

    print("Done. Next: export/convert_to_gguf.py")


if __name__ == "__main__":
    main()
