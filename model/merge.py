#!/usr/bin/env python3
"""
Merge GoLLIE-Plus LoRA adapters with the Llama-3 8B base model.

Install requirements in your conda env first:
    pip install "torch>=2.2" transformers peft bitsandbytes

Run from the same directory:
    cd /sorgin1/users/neildlf/GoLLIE-dev/model
    python merge.py
"""

import logging
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    dtype: str = "bfloat16",
) -> None:
    """Load base weights and LoRA adapters, merge, then save."""
    logging.info("Loading base model: %s", base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else getattr(torch, dtype),
        device_map="auto",          # spreads layers across available GPUs/CPU
    )

    logging.info("Loading LoRA adapters from: %s", lora_path)
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    logging.info("Merging adapters into base weights …")
    merged = lora_model.merge_and_unload()   # returns a plain HF model

    logging.info("Saving merged model to: %s", output_path)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(base_model_path).save_pretrained(output_path)

    logging.info("Done – merged model ready.")


def main() -> None:
    here = Path(__file__).parent

    base_model = "meta-llama/Meta-Llama-3-8B"
    lora_dir   = here / "GUIDEX+Gold_FT" / "GoLLIE+-8b_Llama3_BS128_R128_finetuning"
    out_dir    = here / "GUIDEX+Gold_FT" / "GUIDEX_GOLD_8B"

    merge_lora(
        base_model_path=str(base_model),
        lora_path=str(lora_dir),
        output_path=str(out_dir),
        dtype="bfloat16",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
