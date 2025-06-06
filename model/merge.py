#!/usr/bin/env python3
"""
Merge GUIDEX+Gold_FT LoRA adapters into Meta-Llama-3-8B on CPU and save as
GUIDEX_GOLD_8B.

Run:
    cd /sorgin1/users/neildlf/GoLLIE-dev/model
    python merge.py
"""

from pathlib import Path
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# ── paths ──────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
LORA_DIR   = HERE / "Gold_FT" / "GoLLIE+-8b_Llama3_BS128_R128"
OUT_DIR    = HERE / "Gold_FT" / "GOLD_8B"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.info("Base model : %s", BASE_MODEL)
logging.info("LoRA path  : %s", LORA_DIR)
logging.info("🚀  Everything will be loaded on CPU to avoid GPU OOM.")

# ── 1. load base model fully on CPU ────────────────────────────────────────
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,               # keeps RAM reasonable
    device_map={"": "cpu"},                  # **stay on CPU**
    low_cpu_mem_usage=True,
)

# ── 2. load adapter config & patch target-modules if needed ────────────────
cfg = PeftConfig.from_pretrained(LORA_DIR)
if set(cfg.target_modules) in ({"base_layer"}, set()):
    cfg.target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    logging.info("Patched adapter target-modules → %s", cfg.target_modules)

# ── 3. attach adapters, merge, save ────────────────────────────────────────
lora_model = PeftModel.from_pretrained(base, LORA_DIR, config=cfg)
merged     = lora_model.merge_and_unload()        # plain HF model, still on CPU

merged.save_pretrained(OUT_DIR)
AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(OUT_DIR)
logging.info("✅  merged weights written to  %s", OUT_DIR)
