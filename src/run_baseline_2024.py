"""S0 on the 2024 task, per attribute.

  python src/run_baseline_2024.py [--heuristic cued] [--split val] [--gold]

`dim` is the primary metric here, not `value`: 2024 gold units are normalised
by the challenge (gram, kilogram) while packaging prints whatever it likes
(12 OZ), so a dimensionally correct answer in the wrong unit is a success the
`value` column would score as a failure.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import evaluate as ev
import ocr as O
import rules2024

SHARE = os.path.expanduser("~/Multimodal_Product_Attribution_Extraction_SHARE")
WORKING = f"{SHARE}/data/raw/working_set.csv"
GOLD = f"{SHARE}/data/gold_test_set/review_sheet.csv"


def load_gold_rows():
    for r in csv.DictReader(open(GOLD)):
        yield A.Record2024(sample_id=r["row_uid"], image_link=r["image_link"],
                           entity_name=r["entity_name"],
                           entity_value=r["scraped_entity_value"], group_id="")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heuristic", default="cued", choices=list(rules2024.HEURISTICS) + ["all"])
    ap.add_argument("--split", default=None, choices=["train", "val", "test"])
    ap.add_argument("--gold", action="store_true", help="score the 175-row gold set")
    ap.add_argument("--ws", action="store_true",
                    help="use the rebuilt working set (natural proportions, 8 attributes)")
    args = ap.parse_args()

    if args.ws:
        recs = [r for r in A.load_working_set("data2024/working_set.csv",
                                              split=args.split) if r.gold]
        cache = O.load_cache("cache/ocr_ws2024.jsonl")
        title = f"rebuilt working set{f' ({args.split})' if args.split else ''}"
    elif args.gold:
        recs = [r for r in load_gold_rows() if r.gold]
        cache = O.load_cache("cache/ocr_2024_gold.jsonl")
        title = "175-row gold set"
    else:
        recs = [r for r in A.load(WORKING) if r.gold]
        cache = O.load_cache("cache/ocr_2024.jsonl")
        if args.split:
            assign = A.assign_splits(recs)
            recs = [r for r in recs if assign[r.sample_id] == args.split]
        title = f"working set{f' ({args.split})' if args.split else ''}"

    heuristics = rules2024.HEURISTICS if args.heuristic == "all" else [args.heuristic]
    print(f"S0 on 2024 - {title}, n={len(recs):,}\n")

    for h in heuristics:
        per: dict[str, Counter] = {}
        tot = Counter()
        for r in recs:
            text = O.clean(cache.get(r.image_file, ""))
            pred = rules2024.predict(r.entity_name, text, h)
            s = ev.score(pred, r.gold)
            c = per.setdefault(r.entity_name, Counter())
            c["n"] += 1
            c["abstain"] += pred.abstained
            for k, v in s.items():
                c[k] += v
                tot[k] += v
            tot["n"] += 1
            tot["abstain"] += pred.abstained
        n = tot["n"]
        print(f"--- heuristic: {h} ---")
        hdr = f"{'attribute':<32}{'n':>6}{'cover':>8}{'value':>8}{'dim':>8}{'strict':>8}"
        print(hdr); print("-" * len(hdr))
        for a in sorted(per, key=lambda x: -per[x]["n"]):
            c = per[a]
            print(f"{a:<32}{c['n']:>6,}{100*(c['n']-c['abstain'])/c['n']:>7.1f}%"
                  f"{100*c['value']/c['n']:>7.1f}%{100*c['dim']/c['n']:>7.1f}%"
                  f"{100*c['strict']/c['n']:>7.1f}%")
        print(f"{'ALL':<32}{n:>6,}{100*(n-tot['abstain'])/n:>7.1f}%"
              f"{100*tot['value']/n:>7.1f}%{100*tot['dim']/n:>7.1f}%{100*tot['strict']/n:>7.1f}%\n")


if __name__ == "__main__":
    main()
