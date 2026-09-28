"""Build the S1 training file and report how much of the corpus it covers.

  python src/make_spans.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import evaluate
import label_spans
import ocr as O
import parse
import splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true",
                    help="append the on-package OCR channel (writes spans_ocr_*.jsonl)")
    ap.add_argument("--ocr-cache", default="cache/ocr_train.jsonl")
    args = ap.parse_args()

    cache = O.load_cache(args.ocr_cache) if args.ocr else {}
    prefix = "spans_ocr" if args.ocr else "spans"
    if args.ocr:
        print(f"OCR cache: {len(cache):,} images")

    recs = list(parse.load("data/train.csv"))
    assign = splits.assign(recs)
    by_bucket_total: Counter = Counter()
    by_bucket_ok: Counter = Counter()
    tiers: Counter = Counter()
    bridged: Counter = Counter()
    out: dict[str, list] = {"train": [], "val": [], "test": []}

    for r in recs:
        gold = evaluate.gold_of(r)
        if gold is None:
            continue
        b = evaluate.bucket(r, gold)
        by_bucket_total[b] += 1
        ex = label_spans.build(r, gold, cache.get(r.image_file) if args.ocr else None)
        if ex is None:
            continue
        by_bucket_ok[b] += 1
        tiers[ex["tier"]] += 1
        bridged[ex["bridged"]] += 1
        ex["bucket"] = b
        out[assign[r.sample_id]].append(ex)

    os.makedirs("cache", exist_ok=True)
    for split, rows in out.items():
        n = label_spans.write_jsonl(f"cache/{prefix}_{split}.jsonl", rows)
        print(f"cache/{prefix}_{split}.jsonl  {n:,} examples")

    tot, ok = sum(by_bucket_total.values()), sum(by_bucket_ok.values())
    print(f"\nweak-supervision coverage: {ok:,}/{tot:,} = {100*ok/tot:.1f}%\n")
    print(f"{'bucket':<22}{'labelled':>10}{'of':>9}{'rate':>8}")
    print("-" * 49)
    for b in evaluate.BUCKETS:
        t = by_bucket_total[b]
        if t:
            print(f"{b:<22}{by_bucket_ok[b]:>10,}{t:>9,}{100*by_bucket_ok[b]/t:>7.1f}%")
    print(f"\nounce/fluid-ounce bridge fired on {bridged[True]:,} "
          f"({100*bridged[True]/ok:.1f}%) of labelled rows")
    names = {0: "size alone", 1: "size x explicit pack", 2: "size x bare integer"}
    print("\nassignment tiers (simplest explanation wins):")
    for t, c in sorted(tiers.items()):
        print(f"  tier {t}  {names[t]:<24}{c:>8,}{100*c/ok:>7.1f}%")


if __name__ == "__main__":
    main()
