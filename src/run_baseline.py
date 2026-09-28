"""Run S0 over the corpus and print the stratified report.

  python src/run_baseline.py [--limit N] [--split val]
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate
import parse
import rules
import splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    args = ap.parse_args()

    recs = list(parse.load(args.csv, limit=args.limit))
    print(f"loaded {len(recs):,} records from {args.csv}")

    assign = splits.assign(recs)
    sizes = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "val", "test")}
    ngroups = len(set(splits.group_ids(recs).values()))
    print(f"groups: {ngroups:,} over {len(recs):,} rows "
          f"({len(recs)/ngroups:.2f} rows/group) -> " +
          ", ".join(f"{k} {v:,}" for k, v in sizes.items()))

    rep = evaluate.Report()
    skipped = 0
    for r in recs:
        if args.split and assign[r.sample_id] != args.split:
            continue
        gold = evaluate.gold_of(r)
        if gold is None:
            skipped += 1
            continue
        rep.add(r, rules.predict(r), gold)

    print(f"skipped {skipped:,} rows with unusable gold labels")
    print(rep.table(f"S0 rule baseline - {args.split or 'all rows'}"))


if __name__ == "__main__":
    main()
