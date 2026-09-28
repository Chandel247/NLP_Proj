"""Package the untouched test split for the final dimension-selection evaluation.

Every choice so far - resolution, epochs, the routing rule, the blank-image
control - was made while looking at the 406 validation rows, so those rows can
no longer support a headline number. This builds the same package for the test
split, which nothing has been fitted on: same candidate construction, same
image re-encoding, same CSV columns, so the Kaggle notebook can score it with
the training notebook's eval code unchanged.

Unlike the training package this keeps *every* row - including the ones with no
candidates and the ones where no candidate is correct - because the headline is
end-to-end accuracy over all length rows, not accuracy given a usable list.

    python src/build_test_package.py
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
import ocr as O
from build_lora_package import CAP, Q, candidates_for

LENGTH = ("depth", "width", "height")
STAGE = "kaggle/_test"
OUT = "kaggle/vlm_test_data.tar.gz"


def train_image_names() -> set[str]:
    """Filenames shipped in the training package, to prove none of them recur."""
    if not os.path.exists("kaggle/vlm_lora_data.tar.gz"):
        return set()
    with tarfile.open("kaggle/vlm_lora_data.tar.gz") as t:
        return {os.path.basename(n) for n in t.getnames()
                if n.startswith("vlm_lora/images/") and n.endswith(".jpg")}


def main() -> None:
    cache = O.load_cache("cache/ocr_ws2024.jsonl")
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(f"{STAGE}/images", exist_ok=True)

    # The split key is group_id, but a photo can be listed under two group_ids:
    # 41Ol0KtcHdL.jpg is group 347320 (train, depth) and 558806 (test, height).
    # One row, and the model was trained on that exact image, so it comes out.
    seen = train_image_names()

    rows, wanted, dropped = [], {}, 0
    for r in A.load_working_set("data2024/working_set.csv", split="test"):
        if not r.gold or r.entity_name not in LENGTH:
            continue
        path = O.image_path(r.image_file)
        if not path:
            continue
        if r.image_file in seen:
            dropped += 1
            continue
        uniq, gold_idx = candidates_for(r, cache)
        lo, hi, gu = r.gold
        rows.append([r.image_file, r.entity_name,
                     "; ".join(f"{v:g} {u}" for v, u in uniq),
                     gold_idx[0] if gold_idx else -1, lo, hi, gu])
        wanted[r.image_file] = path

    with open(f"{STAGE}/test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_file", "attribute", "candidates",
                    "gold_index", "gold_lo", "gold_hi", "gold_unit"])
        w.writerows(rows)

    overlap = seen & set(wanted)
    assert not overlap, f"training images still present: {sorted(overlap)}"
    print(f"rows: {len(rows):,}  images: {len(wanted):,}  "
          f"(dropped {dropped} row(s) whose photo appears in training)")
    n0 = sum(1 for r in rows if not r[2])
    n1 = sum(1 for r in rows if r[2] and ";" not in r[2])
    print(f"no candidates: {n0:,}  single candidate: {n1:,}  "
          f"gold among candidates: {sum(1 for r in rows if r[3] >= 0):,}")

    print(f"re-encoding {len(wanted):,} images at {CAP}px q{Q} ...", flush=True)
    for i, (name, path) in enumerate(wanted.items(), 1):
        im = Image.open(path).convert("RGB")
        im.thumbnail((CAP, CAP), Image.LANCZOS)
        im.save(f"{STAGE}/images/{name}", "JPEG", quality=Q, optimize=True)
        if i % 500 == 0:
            print(f"  {i:,}/{len(wanted):,}", flush=True)

    shutil.copy2("src/normalize.py", f"{STAGE}/normalize.py")
    with tarfile.open(OUT, "w:gz") as t:
        t.add(STAGE, arcname="vlm_test")
    shutil.rmtree(STAGE)
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1048576:.0f} MB)")


if __name__ == "__main__":
    main()
