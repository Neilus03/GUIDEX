#!/usr/bin/env python3
"""
GUIDEX pipeline (final, complete)
================================

Generates GUIDEX‑style annotations and outputs JSONL identical in structure to the
files produced by `vllm_simple_dataset_gen_batched.py`.

Each line looks like:
```json
{
  "ids": ["<uuid>"] ,
  "task_id": "fineweb-edu_pretrain_gollie2",
  "scorer_cls": "src.tasks.fineweb-edu_pretrain.scorer.fineweb-edu_pretrainScorer",
  "labels": "result_instances = [ … ]",      # python list string
  "text":   "# The following lines describe the task definition\n@dataclass …",  # dataclass code string
  "unlabelled_sentence": "original passage"
}
```

Three stages
------------
1. **annotate**   – 4‑step chat (summary → compact JSON → dataclass guidelines → instances).
2. **dedup**      – remove duplicates by `unlabelled_sentence`.
3. **exec_filter** – keep only rows where `text + "\n" + labels` executes without error.

Run
---
```bash
python GUIDEX_pipeline.py --input fineweb.json --output guidex_out.jsonl --batch-size 32
```
"""
from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
from time import time
from typing import Any, Dict, List

import torch
from huggingface_hub import login
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os

# ─────────────────── configuration ────────────────────
MODEL_NAME         = "meta-llama/Llama-3.1-70B-Instruct"

# Load the Hugging Face token from an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

CACHE_DIR          = str(Path.home() / ".cache/huggingface")
RAW_DIR            = Path("raw_outputs")
LOG_FILE           = "guidex_pipeline.log"
SAMPLING_KWARGS    = dict(temperature=0.7, top_p=0.95, max_tokens=1024)
DEFAULT_BATCH_SIZE = 32
MAX_HISTORY_TURNS  = 4

TASK_ID    = "fineweb-edu_pretrain_gollie2"
SCORER_CLS = "src.tasks.fineweb-edu_pretrain.scorer.fineweb-edu_pretrainScorer"

# ─────────────────── logging ───────────────────────────
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)

# ─────────────────── model ────────────────────────────
login(token=HF_TOKEN)
torch.cuda.empty_cache()
logging.info("Loading model %s …", MODEL_NAME)
llm = LLM(model=MODEL_NAME, tokenizer=MODEL_NAME, dtype=torch.bfloat16,
          download_dir=CACHE_DIR, tensor_parallel_size=2, max_model_len=8192)
logging.info("Model ready")

# ─────────────────── helpers ───────────────────────────
RAW_DIR.mkdir(exist_ok=True)

_DEF_PROMPT_STEP2 = (
    """Based on the extracted summary and the original text, synthesize the information into a JSON output. Keep it as less verbose as possible. The strings in the json must match the original text. The name of each key should be properly chosen and general, similar fields should be merged. You should populate all the attributes and be as concise as possible. The attributes can't be populated with full sentences; they must be as short as possible. Remember that it should match the contents of text accurately. The only thing that you should return is a single json that contains the required content, nothing else, no text introducing the json, no text saying how you hope it is ok. JUST THE JSON."""
)

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _append_jsonl(obj: Dict[str, Any], path: Path):
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _save_raw(txt: str, fname: str):
    with (RAW_DIR / fname).open("a", encoding="utf-8") as f:
        f.write(txt + "\n\n")

def _extract_code(block: str) -> str:
    m = re.search(r"```python(.*?)```", block, re.DOTALL)
    return (m.group(1) if m else block).strip()

def _build_prompts(cur: List[str], hist: List[List[Dict[str, str]]]):
    out = []
    for p, h in zip(cur, hist):
        if h:
            joined = "".join(f"{e['role'].capitalize()}: {e['content']}\n" for e in h)
            out.append(joined + f"User: {p}\nAssistant:")
        else:
            out.append(f"User: {p}\nAssistant:")
    return out

def _query_llm(prompts: List[str], hist: List[List[Dict[str, str]]]):
    outputs = llm.generate(_build_prompts(prompts, hist), SamplingParams(**SAMPLING_KWARGS))
    return [o.outputs[0].text.strip() for o in outputs]

def _push(hist, prompt, resp, end=False):
    if end:
        return []
    hist.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}])
    if len(hist) > MAX_HISTORY_TURNS * 2:
        hist[:] = hist[-MAX_HISTORY_TURNS * 2:]
    return hist

def _make_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ids": [doc.get("id", "")],
        "task_id": TASK_ID,
        "scorer_cls": SCORER_CLS,
        "labels": doc["result_instances"].strip(),          # list-only string
        "text": (
            "# The following lines describe the task definition\n"
            + doc["annotation_guidelines_as_dataclass"].strip()
            + "\n\n# This is the text to analyze\ntext = "
            + json.dumps(doc["text"])
            + "\n\n# The list called result contains the instances for the following events according to the guidelines above:\n"
            + "result = " + doc["result_instances"].strip()
        ),
        "unlabelled_sentence": doc["text"].strip(),
    }

# ───────── prompts P3 / P4 ─────────
PROMPT_P3 = '''Based on the JSON output, the summary and the original text, generate annotation guidelines that include 1. a high quality, complete, and extensive description that is long enough, and 2. the expected format for each field. Then turn this annotations into a python file consisting of python dataclass objects. Description of the class should be given as a docstring and descriptions of the attributes must be given as comments, without examples. Create a class that wraps all the information together. Return only one python file that contains this annotation guidelines. The kind of output I am expecting is similar (but not limited) to this, but substituting the placeholders with the specifics of the text being discussed, and creating as many classes as needed and naming them properly:
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EntityA:
    """
    A generic but long enough description for EntityA, explaining its purpose and characteristics
    in a general context without specific details.
    """
    identifier: str
    """
    A brief explanation of what the identifier represents in a general sense.
    """
    attribute1: str # A short comment explaining this attribute generically
    attribute2: List[str]
    """
    A multi-line comment explaining what this list typically contains
    and its significance to EntityA.
    # Add more attributes as needed
    """

@dataclass
class EntityB:
    """
    A generic description for EntityB, explaining its purpose and characteristics
    in a general context without specific details.
    """
    mention: str
    """
    An explanation of what the mention represents for EntityB.
    """
    date: str # A comment about what this date signifies
    location: str # A comment about what this location represents
    element: str # A comment about what this element represents
    # Add more attributes as needed
```'''

PROMPT_P4 = '''Based on the annotation guidelines, you will provide a python list called result_instances containing instances of the dataclasses based on the information on the text. Be as concise as possible when extracting the instances, which should be single words or numbers when possible, not longer. You should only provide the python result_instances list with the instances, nothing else. Do not provide explanations nor extra data apart from what's contained in the result_instances list. The kind of output I am expecting is similar (but not limited) to this, but substituting the placeholders with the specifics of the text being discussed:
```python
result_instances = [
    EntityB(name="EntityB_Name1", date="Date1", location1="Location1A", element="element1"),
    EntityA(identifier="EntityA_ID1", attribute1="AttributeValue1", attribute2=["Item1", "Item2", "Item3"])
]
```'''

# ──────────── Stage 1 ───────────

def annotate(docs: List[Dict[str, Any]], dest: Path, batch: int):
    logging.info("[1/3] Annotating %d docs (batch=%d)…", len(docs), batch)
    for start in tqdm(range(0, len(docs), batch), unit="batch"):
        chunk = docs[start:start + batch]
        hist = [[] for _ in chunk]

        # 1 summary
        p1 = [
            f"Use bullet points to summarize the main ideas of the following text, keeping the most important information. Text: '{d['text']}'. Summarize these points concisely."
            for d in chunk
        ]

        r1 = _query_llm(p1, hist)
        for i, r in enumerate(r1):
            _save_raw(r, f"step1_{start+i}.txt")
            hist[i] = _push(hist[i], p1[i], r)

        # 2 compact JSON
        p2 = [_DEF_PROMPT_STEP2 for _ in chunk]
        r2 = _query_llm(p2, hist)
        for i, r in enumerate(r2):
            _save_raw(r, f"step2_{start+i}.txt")
            hist[i] = _push(hist[i], _DEF_PROMPT_STEP2, r)

        # 3 guidelines
        r3 = _query_llm([PROMPT_P3] * len(chunk), hist)
        guidelines = []
        for i, r in enumerate(r3):
            _save_raw(r, f"step3_{start+i}.txt")
            hist[i] = _push(hist[i], PROMPT_P3, r)
            guidelines.append(_extract_code(r))

        # 4 result_instances
        r4 = _query_llm([PROMPT_P4] * len(chunk), hist)
        instances = []
        for i, r in enumerate(r4):
            _save_raw(r, f"step4_{start+i}.txt")
            hist[i] = _push(hist[i], PROMPT_P4, r, end=True)
            raw = _extract_code(r)
            
            cleaned = raw.lstrip()                          
            if cleaned.startswith("result_instances"):     
                cleaned = cleaned.split("=", 1)[1].lstrip() 
            instances.append(cleaned)

        # write
        for i, doc in enumerate(chunk):
            doc["annotation_guidelines_as_dataclass"] = guidelines[i]
            doc["result_instances"] = instances[i]
            if guidelines[i] and instances[i]:
                _append_jsonl(_make_record(doc), dest)
            else:
                logging.warning("[skip] incomplete doc %d", start + i)

# ──────────── Stage 2: dedup ───────────

def dedup(src: Path, dst: Path):
    seen = set(); kept = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            obj = json.loads(line)
            sent = obj["unlabelled_sentence"]
            if sent not in seen:
                seen.add(sent)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    logging.info("[2/3] Deduplicated – kept %d unique rows", kept)

# ──────────── Stage 3: exec filter ───────────

def exec_filter(src: Path, dst: Path):
    gbl = {"List": list, "Optional": type(None), "Dict": dict, "Any": object}
    kept = 0
    with src.open() as fin, dst.open("w") as fout:
        for lno, line in enumerate(fin, 1):
            obj = json.loads(line)
            code = obj["text"] + "\n" + obj["labels"]
            try:
                exec(code, gbl)
            except Exception as e:
                logging.debug("exec error line %d: %s", lno, e)
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    logging.info("[3/3] Exec‑filter kept %d rows", kept)

# ──────────── load input ───────────

def load_input(path: Path, limit: int | None):
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        rows = json.loads(path.read_text())["items"]
    return rows[:limit] if limit else rows

# ──────────── main ───────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--subset", type=int)
    args = ap.parse_args()

    t0 = time()
    inp = Path(args.input); out = Path(args.output)

    stage1 = out.with_suffix(".jsonl")
    if not stage1.exists():
        annotate(load_input(inp, args.subset), stage1, args.batch_size)
    else:
        logging.info("[skip] annotation stage – %s exists", stage1)

    stage2 = out.with_suffix(".dedup.jsonl")
    if not stage2.exists():
        dedup(stage1, stage2)
    else:
        logging.info("[skip] dedup stage – %s exists", stage2)

    exec_filter(stage2, out)
    logging.info("Finished → %s  (%.1fs)", out, time() - t0)

if __name__ == "__main__":
    main()
