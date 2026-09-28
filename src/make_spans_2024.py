"""Weak-supervision spans for the rebuilt 2024 working set.

The program search runs backwards from the gold value, which is what makes it
usable on the length attributes: a photo reading "12 x 8 x 3 cm" offers three
candidates, and knowing the answer is 8 tells us *which span* to label. The
tagger then learns to make that choice from the attribute name and surrounding
words - the choice S0 gets wrong 54% of the time.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import label_spans
import ocr as O
import rules2024

LENGTH = {"depth", "width", "height"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data2024/working_set.csv")
    ap.add_argument("--ocr", default="cache/ocr_ws2024.jsonl")
    ap.add_argument("--prefix", default="spans2024")
    args = ap.parse_args()

    cache = O.load_cache(args.ocr)
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    tot, ok = Counter(), Counter()
    split_of = {}
    import csv as _csv
    with open(args.csv) as fh:
        for row in _csv.DictReader(fh):
            split_of[row["sample_id"]] = row["split"]

    for r in A.load_working_set(args.csv):
        gold = r.gold
        if gold is None:
            continue
        a = r.entity_name
        tot[a] += 1
        ex = label_spans.build(r, gold, cache.get(r.image_file))
        if ex is None:
            continue
        ok[a] += 1
        ex["attribute"] = a
        out[split_of[r.sample_id]].append(ex)

    for split, rows in out.items():
        n = label_spans.write_jsonl(f"cache/{args.prefix}_{split}.jsonl", rows)
        print(f"cache/{args.prefix}_{split}.jsonl  {n:,} examples")

    n, k = sum(tot.values()), sum(ok.values())
    print(f"\nweak-supervision coverage: {k:,}/{n:,} = {100*k/n:.1f}%\n")
    hdr = f"{'attribute':<32}{'labelled':>10}{'of':>9}{'rate':>8}"
    print(hdr); print("-" * len(hdr))
    for a in sorted(tot, key=lambda x: -tot[x]):
        print(f"{a:<32}{ok[a]:>10,}{tot[a]:>9,}{100*ok[a]/tot[a]:>7.1f}%")
    for label, sel in (("length", LENGTH), ("other", set(tot) - LENGTH)):
        sn = sum(tot[a] for a in sel); sk = sum(ok[a] for a in sel)
        print(f"\n{label:<10}{sk:>10,}{sn:>9,}{100*sk/sn:>7.1f}%")


if __name__ == "__main__":
    main()
