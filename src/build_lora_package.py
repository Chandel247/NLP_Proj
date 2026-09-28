"""Package the LoRA training set for the dimension-selection task.

Target format: given the image and the OCR candidate list, output the letter of
the correct candidate. That is the exact decision the zero-shot model failed -
it picked (b) or (c) 72% of the time regardless of content - and it comes with
free supervision, since weak supervision already knows which candidate matches
the gold value.

Images are re-encoded to 768px, which is the processor's `max_pixels`, so the
model sees no less than it would from the originals at a third of the bytes.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tarfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import evaluate as ev
import normalize as N
import ocr as O
import rules2024 as R

LENGTH = ("depth", "width", "height")
CAP, Q = 768, 85
STAGE = "kaggle/_lora"


def candidates_for(rec, cache):
    """Deduped same-family candidates in OCR order, plus which are correct."""
    lo, hi, gu = rec.gold
    raw = R.candidates(O.clean(cache.get(rec.image_file, "")), "length")
    seen, uniq = set(), []
    for v, u, _ in raw:
        k = round(N.to_base(v, u) or 0, 4)
        if k and k not in seen:
            seen.add(k)
            uniq.append((v, u))
    uniq = uniq[:8]
    gold_idx = [i for i, (v, u) in enumerate(uniq)
                if ev._in_range(N.catalog_convert(v, u, gu)[0] or -1, lo, hi)]
    return uniq, gold_idx


def main() -> None:
    cache = O.load_cache("cache/ocr_ws2024.jsonl")
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(f"{STAGE}/images", exist_ok=True)

    eval_ids = set()
    if os.path.exists("kaggle/vlm_select_results.json"):
        eval_ids = {r["image_file"] for r in json.load(open("kaggle/vlm_select_results.json"))}
        print(f"reusing the {len(eval_ids)} round-2 eval rows so results stay comparable")

    wanted, counts = {}, {"train": 0, "eval": 0}
    writers = {}
    fhs = {}
    for name in ("train", "eval"):
        fhs[name] = open(f"{STAGE}/{name}.csv", "w", newline="")
        writers[name] = csv.writer(fhs[name])
        writers[name].writerow(["image_file", "attribute", "candidates",
                                "gold_index", "gold_lo", "gold_hi", "gold_unit"])

    for split in ("train", "val"):
        for r in A.load_working_set("data2024/working_set.csv", split=split):
            if not r.gold or r.entity_name not in LENGTH:
                continue
            if not O.image_path(r.image_file):
                continue
            # Eval keeps the exact round-2 rows; training takes the train split
            # only, so nothing leaks across.
            dest = "eval" if r.image_file in eval_ids else ("train" if split == "train" else None)
            if dest is None:
                continue
            uniq, gold_idx = candidates_for(r, cache)
            if dest == "train" and not (len(uniq) >= 2 and gold_idx):
                continue
            lo, hi, gu = r.gold
            writers[dest].writerow([
                r.image_file, r.entity_name,
                "; ".join(f"{v:g} {u}" for v, u in uniq),
                gold_idx[0] if gold_idx else -1, lo, hi, gu])
            counts[dest] += 1
            wanted[r.image_file] = O.image_path(r.image_file)

    for fh in fhs.values():
        fh.close()

    print(f"rows: train {counts['train']:,}  eval {counts['eval']:,}")
    print(f"re-encoding {len(wanted):,} images at {CAP}px q{Q} ...", flush=True)
    for i, (name, path) in enumerate(wanted.items(), 1):
        im = Image.open(path).convert("RGB")
        im.thumbnail((CAP, CAP), Image.LANCZOS)
        im.save(f"{STAGE}/images/{name}", "JPEG", quality=Q, optimize=True)
        if i % 1500 == 0:
            print(f"  {i:,}/{len(wanted):,}", flush=True)

    shutil.copy2("src/normalize.py", f"{STAGE}/normalize.py")
    out = "kaggle/vlm_lora_data.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        t.add(STAGE, arcname="vlm_lora")
    shutil.rmtree(STAGE)
    print(f"\nwrote {out}  ({os.path.getsize(out)/1048576:.0f} MB)")


if __name__ == "__main__":
    main()
