"""Re-score every system once labels are corrected.

Two modes, because one is available today and the other waits on a human:

  --exclude-wrong   drop the rows `label_audit.py` proves wrong. Needs no
                    verification at all, so it can run right now.
  --verified        score only rows with a human-filled `verified_value`,
                    reporting n and a Wilson interval so partial progress is
                    still usable.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import evaluate as ev
import ocr as O
import rules2024
import verify_labels as V


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - honest at small n, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-wrong", action="store_true")
    ap.add_argument("--verified", action="store_true")
    args = ap.parse_args()

    wrong = set()
    if os.path.exists("gold/provably_wrong_gold.csv"):
        with open("gold/provably_wrong_gold.csv") as fh:
            wrong = {r["row_uid"] for r in csv.DictReader(fh)}

    verified: dict[str, str] = {}
    path = "gold/review_sheet_enriched.csv"
    if args.verified and os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r.get("verified_value", "").strip():
                    verified[r["row_uid"]] = r["verified_value"].strip()

    rows = V.build_rows()
    kept, dropped_wrong, dropped_unverified = [], 0, 0
    for row in rows:
        sid = row.rec.sample_id
        if args.exclude_wrong and sid in wrong:
            dropped_wrong += 1
            continue
        if args.verified:
            if sid not in verified:
                dropped_unverified += 1
                continue
            g = A.parse_entity_value(verified[sid])
            if g is None:
                dropped_unverified += 1
                continue
            row.rec.entity_value = verified[sid]
        kept.append(row)

    print(f"gold rows: {len(rows)}")
    if args.exclude_wrong:
        print(f"  dropped as provably wrong : {dropped_wrong}")
    if args.verified:
        print(f"  dropped as not yet verified: {dropped_unverified}")
    print(f"  scored                     : {len(kept)}\n")
    if not kept:
        print("nothing to score — fill in verified_value, or run --exclude-wrong")
        return

    systems = {
        "S0 rules + Apple Vision": lambda r: r.ours,
        "their pipeline (EasyOCR)": lambda r: r.theirs,
        "their gpt-4o-mini": lambda r: r.llm,
    }
    hdr = f"{'system':<26}{'n':>5}{'acc':>8}{'95% CI':>16}{'cover':>8}{'prec|ans':>10}"
    print(hdr); print("-" * len(hdr))
    for name, get in systems.items():
        n = len(kept)
        answered = sum(get(r) is not None for r in kept)
        ok = sum(V.agrees(get(r), r.rec.gold) for r in kept)
        lo, hi = wilson(ok, n)
        prec = 100 * ok / answered if answered else 0.0
        print(f"{name:<26}{n:>5}{100*ok/n:>7.1f}%"
              f"{f'[{100*lo:.1f}, {100*hi:.1f}]':>16}{100*answered/n:>7.1f}%{prec:>9.1f}%")


if __name__ == "__main__":
    main()
