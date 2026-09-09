"""
[RUNS: COLAB — CPU is fine]

The GGUF bridge. This is the single most important hop in the whole project:
it turns a cloud-trained HuggingFace model into a file your laptop can run.

Two stages, both from llama.cpp:
  1. convert_hf_to_gguf.py  -> merged fp16 model  ->  an fp16 .gguf
  2. llama-quantize          -> fp16 .gguf         ->  a small q4_k_m .gguf

The quantized .gguf is what you download and hand to Ollama.

This script shells out to llama.cpp rather than reimplementing it. It clones and
builds llama.cpp on first run (a few minutes in Colab).
"""

import subprocess
import sys
from pathlib import Path
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def ensure_llama_cpp(root=Path("llama.cpp")):
    """Clone + build llama.cpp if not already present."""
    if not root.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", str(root)])
    # Build the quantize tool. On Colab, cmake is available.
    build = root / "build"
    if not (build / "bin" / "llama-quantize").exists():
        run(["cmake", "-B", str(build), "-S", str(root)])
        run(["cmake", "--build", str(build), "--config", "Release",
             "-j", "--target", "llama-quantize"])
    # Install python deps the converter needs.
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         str(root / "requirements.txt")])
    return root


def main():
    cfg = load_config()
    merged_dir = cfg["model"]["merged_dir"]
    quant = cfg["export"]["quant"]
    gguf_out = Path(cfg["export"]["gguf_out"])
    gguf_out.parent.mkdir(parents=True, exist_ok=True)

    llama = ensure_llama_cpp()
    fp16_gguf = gguf_out.with_name("model-fp16.gguf")

    # Stage 1: HF -> fp16 GGUF
    run([sys.executable, str(llama / "convert_hf_to_gguf.py"),
         merged_dir, "--outfile", str(fp16_gguf), "--outtype", "f16"])

    # Stage 2: fp16 GGUF -> quantized GGUF
    quantize_bin = llama / "build" / "bin" / "llama-quantize"
    run([str(quantize_bin), str(fp16_gguf), str(gguf_out), quant])

    size_mb = gguf_out.stat().st_size / 1e6
    print(f"\nGGUF ready: {gguf_out} ({size_mb:.0f} MB, {quant})")
    print("Download this file, drop it next to ollama/Modelfile, then run ollama/import.sh")


if __name__ == "__main__":
    main()
