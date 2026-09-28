"""Zero-shot VLM probe on the three dimension attributes.

This is the one component not bounded by the 71.7% OCR ceiling: the OCR pipeline
can read `9.3cm` but nothing in the text says it is the *width*, so the cascade
stalls at 36.2%. A VLM sees the arrow and the axis it annotates, which is where
that information actually lives.

Probe before committing, as with the OCR pass: a small sample here decides
whether a LoRA fine-tune on Colab is worth the session.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import evaluate as ev
import normalize as N
import ocr as O
import rules2024
import verify_labels as V

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
LENGTH = ("depth", "width", "height")

PHRASE = {"depth": "depth (front to back)",
          "width": "width (left to right)",
          "height": "height (top to bottom)"}

PROMPT = (
    "This is a product photograph, which may include printed dimension labels or arrows.\n"
    "What is the product's {phrase}?\n"
    "Answer with only a number and a unit, for example \"12 cm\" or \"4.5 in\".\n"
    "If the {attr} is not shown, answer exactly \"unknown\"."
)

_ANS = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*([A-Za-z\"'.]+)")


def parse_answer(raw: str) -> tuple[float, str] | None:
    """Pull (value, canonical unit) out of free-form model text."""
    if not raw or "unknown" in raw.lower():
        return None
    m = _ANS.search(raw.replace("″", "in").replace("”", "in"))
    if not m:
        return None
    unit = N.normalize_unit(m.group(2).strip("\"'."))
    if unit is None or N.family(unit) != "length":
        return None
    return float(m.group(1).replace(",", ".")), unit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--out", default="cache/vlm_probe.jsonl")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    recs = [r for r in A.load_working_set("data2024/working_set.csv", split=args.split)
            if r.gold and r.entity_name in LENGTH and O.image_path(r.image_file)]
    random.seed(args.seed)
    sample = random.sample(recs, min(args.n, len(recs)))
    print(f"{len(recs):,} length rows in {args.split}; probing {len(sample)}", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL} on {dev} ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28,
                                         max_pixels=args.max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16).to(dev).eval()
    print("loaded", flush=True)

    cache = O.load_cache("cache/ocr_ws2024.jsonl")
    hits = Counter()
    rows = []
    t0 = time.time()
    for i, r in enumerate(sample):
        img = Image.open(O.image_path(r.image_file)).convert("RGB")
        prompt = PROMPT.format(phrase=PHRASE[r.entity_name], attr=r.entity_name)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        raw = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip()

        pred = parse_answer(raw)
        from rules import Prediction, ABSTAIN
        p_vlm = ABSTAIN if pred is None else Prediction(pred[0], pred[1], "vlm")
        p_s0 = rules2024.predict(r.entity_name, O.clean(cache.get(r.image_file, "")))

        s_vlm = ev.score(p_vlm, r.gold)
        s_s0 = ev.score(p_s0, r.gold)
        hits["n"] += 1
        hits["vlm_answered"] += not p_vlm.abstained
        hits["vlm_dim"] += s_vlm["dim"]
        hits["s0_answered"] += not p_s0.abstained
        hits["s0_dim"] += s_s0["dim"]
        hits[f"{r.entity_name}_n"] += 1
        hits[f"{r.entity_name}_vlm"] += s_vlm["dim"]
        hits[f"{r.entity_name}_s0"] += s_s0["dim"]
        rows.append({"sample_id": r.sample_id, "attribute": r.entity_name,
                     "gold": list(r.gold), "vlm_raw": raw,
                     "vlm_pred": None if pred is None else list(pred),
                     "vlm_dim": s_vlm["dim"], "s0_dim": s_s0["dim"]})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(sample)}  {el/(i+1):.1f}s/img  "
                  f"vlm {100*hits['vlm_dim']/hits['n']:.1f}%  "
                  f"s0 {100*hits['s0_dim']/hits['n']:.1f}%", flush=True)

    with open(args.out, "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")

    n = hits["n"]
    print(f"\nprobed {n} rows in {(time.time()-t0)/60:.1f} min\n")
    hdr = f"{'system':<18}{'coverage':>10}{'dim':>8}"
    print(hdr); print("-" * len(hdr))
    print(f"{'VLM zero-shot':<18}{100*hits['vlm_answered']/n:>9.1f}%{100*hits['vlm_dim']/n:>7.1f}%")
    print(f"{'S0 rules (OCR)':<18}{100*hits['s0_answered']/n:>9.1f}%{100*hits['s0_dim']/n:>7.1f}%")
    print(f"\n{'attribute':<12}{'n':>5}{'VLM':>8}{'S0':>8}")
    print("-" * 33)
    for a in LENGTH:
        k = hits[f"{a}_n"] or 1
        print(f"{a:<12}{hits[f'{a}_n']:>5}{100*hits[f'{a}_vlm']/k:>7.1f}%{100*hits[f'{a}_s0']/k:>7.1f}%")
    print(f"\nreference: cascade 36.2% dim on the full val length set; "
          f"OCR ceiling 71.7% (the VLM is not bound by it)")
    print(f"cached -> {args.out}")


if __name__ == "__main__":
    main()
