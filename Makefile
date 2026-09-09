# Convenience targets. Local ones run anywhere; GPU ones are meant for Colab.
.PHONY: prep train merge gguf import eval clean

prep:      ## [LOCAL] format raw data -> chat-template JSONL
	python data/prepare_data.py

train:     ## [COLAB] QLoRA fine-tune
	python train/train_qlora.py

merge:     ## [COLAB] merge adapter -> fp16 model
	python export/merge_adapter.py

gguf:      ## [COLAB] convert + quantize -> GGUF
	python export/convert_to_gguf.py

import:    ## [LOCAL] register GGUF with Ollama and run it
	bash ollama/import.sh

eval:      ## [LOCAL] compare base vs fine-tune via Ollama
	python eval/compare.py

clean:     ## remove generated artifacts
	rm -rf artifacts/ llama.cpp/ __pycache__/
