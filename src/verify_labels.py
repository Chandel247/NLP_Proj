"""Turn provisional scraped labels into a prioritised verification job.

Every metric on this project - ours and the existing pipeline's - is scored
against scraped `entity_value` that nobody has checked.  Rather than eyeball 175
images from scratch, this corroborates each label against four independent
sources and flags only the rows where they fail to agree.

Corroboration is strong evidence, not proof: a package printing both net and
gross weight can corroborate the wrong figure, which is why the auto-accepted
pile still gets a random spot-check (see `--spotcheck`).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import evaluate as ev
import normalize as N
import ocr as O
import rules2024

SHARE = os.path.expanduser("~/Multimodal_Product_Attribution_Extraction_SHARE")
GOLD_DIR = f"{SHARE}/data/gold_test_set"

ABSTAIN_STRINGS = {"", "cannot determine", "none", "nan", "n/a", "unknown"}
_PRED = re.compile(r"(-?\d+(?:\.\d+)?)\s*(.*)")


def parse_prediction(raw: str) -> tuple[float, str] | None:
    """'175 ml' / '1.5 kilogram' -> (value, canonical unit); None if abstained."""
    if raw is None or raw.strip().lower() in ABSTAIN_STRINGS:
        return None
    m = _PRED.match(raw.strip())
    if not m:
        return None
    unit = N.normalize_unit(m.group(2)) if m.group(2).strip() else None
    return None if unit is None else (float(m.group(1)), unit)


def agrees(pred: tuple[float, str] | None, gold: tuple[float, float, str]) -> bool:
    """Dimension-aware: 12 ounce agrees with a gold of 340 gram."""
    if pred is None:
        return False
    lo, hi, gu = gold
    if N.family(pred[1]) != N.family(gu):
        return False
    conv, _ = N.catalog_convert(pred[0], pred[1], gu)
    return conv is not None and ev._in_range(conv, lo, hi)


@dataclass
class Row:
    rec: A.Record2024
    ocr_raw: str
    ours: tuple[float, str] | None = None
    theirs: tuple[float, str] | None = None
    llm: tuple[float, str] | None = None
    ocr_supports: bool = False
    ocr_same_family: list = field(default_factory=list)
    flag: str = ""
    score: int = 0

    @property
    def disputed(self) -> bool:
        """Our answer and theirs both commit, and they differ."""
        if self.ours is None or self.theirs is None:
            return False
        a = N.to_base(*self.ours)
        b = N.to_base(*self.theirs)
        if a is None or b is None or N.family(self.ours[1]) != N.family(self.theirs[1]):
            return True
        return abs(a - b) / max(b, 1e-9) > 0.01


def load_side_predictions() -> tuple[dict, dict]:
    pipe, llm = {}, {}
    with open(f"{GOLD_DIR}/pipeline_predictions.csv") as fh:
        for r in csv.DictReader(fh):
            pipe[r["row_uid"]] = r.get("pred_value", "")
    with open(f"{GOLD_DIR}/llm_baseline_predictions.csv") as fh:
        for r in csv.DictReader(fh):
            llm[r["row_uid"]] = r.get("llm_pred_value", "")
    return pipe, llm


def build_rows() -> list[Row]:
    cache = O.load_cache("cache/ocr_2024_gold.jsonl")
    pipe, llm = load_side_predictions()
    rows = []
    with open(f"{GOLD_DIR}/review_sheet.csv") as fh:
        for r in csv.DictReader(fh):
            rec = A.Record2024(sample_id=r["row_uid"], image_link=r["image_link"],
                               entity_name=r["entity_name"],
                               entity_value=r["scraped_entity_value"], group_id="")
            gold = rec.gold
            if gold is None:
                continue
            raw = cache.get(rec.image_file, "")
            text = O.clean(raw)
            row = Row(rec=rec, ocr_raw=raw)

            pred = rules2024.predict(rec.entity_name, text)
            row.ours = None if pred.abstained else (pred.value, pred.unit)
            row.theirs = parse_prediction(pipe.get(rec.sample_id, ""))
            row.llm = parse_prediction(llm.get(rec.sample_id, ""))

            fam = N.family(gold[2])
            row.ocr_same_family = [(q.value, q.unit) for q in N.find_quantities(text)
                                   if N.family(q.unit) == fam]
            row.ocr_supports = any(agrees(c, gold) for c in row.ocr_same_family)

            row.score = (row.ocr_supports + agrees(row.ours, gold)
                         + agrees(row.theirs, gold) + agrees(row.llm, gold))

            if row.ocr_supports and row.score >= 2:
                row.flag = "corroborated"
            elif row.ocr_same_family and not row.ocr_supports:
                row.flag = "contradicted"
            elif row.disputed:
                row.flag = "disputed"
            elif not row.ocr_same_family:
                row.flag = "unsupported"
            else:
                row.flag = "weak"
            rows.append(row)
    return rows


PRIORITY = ["contradicted", "disputed", "unsupported", "weak", "corroborated"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gold/review_sheet_enriched.csv")
    ap.add_argument("--spotcheck", type=int, default=25,
                    help="corroborated rows to sample for a false-accept estimate")
    args = ap.parse_args()

    rows = build_rows()
    by_flag = Counter(r.flag for r in rows)
    n = len(rows)

    print(f"gold rows with parseable labels: {n}\n")
    hdr = f"{'triage bucket':<16}{'n':>5}{'share':>8}   {'meaning'}"
    print(hdr); print("-" * 74)
    meaning = {
        "contradicted": "package shows a same-family value that DISAGREES",
        "disputed":     "our answer and theirs both commit and differ",
        "unsupported":  "no same-family value anywhere in the OCR",
        "weak":         "package supports it but no system agrees",
        "corroborated": "on package AND >=1 system agrees -> auto-accept",
    }
    for f in PRIORITY:
        c = by_flag.get(f, 0)
        print(f"{f:<16}{c:>5}{100*c/n:>7.1f}%   {meaning[f]}")

    needs_human = sum(by_flag.get(f, 0) for f in PRIORITY[:4])
    print(f"\nneeds human judgement : {needs_human}/{n}  ({100*needs_human/n:.0f}%)")
    print(f"auto-accepted         : {by_flag.get('corroborated',0)}"
          f"  (spot-check {args.spotcheck} to estimate false accepts)")

    print(f"\ncorroboration score distribution (0-4):")
    for s, c in sorted(Counter(r.score for r in rows).items()):
        print(f"  {s}  {c:>4}  {100*c/n:>5.1f}%  {'#'*int(40*c/n)}")

    print(f"\nby attribute:")
    per = {}
    for r in rows:
        per.setdefault(r.rec.entity_name, Counter())[r.flag] += 1
    print(f"  {'attribute':<32}" + "".join(f"{f[:6]:>9}" for f in PRIORITY))
    for a, c in sorted(per.items()):
        print(f"  {a:<32}" + "".join(f"{c.get(f,0):>9}" for f in PRIORITY))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    order = {f: i for i, f in enumerate(PRIORITY)}
    rows.sort(key=lambda r: (order[r.flag], -len(r.ocr_same_family)))
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["review_order", "row_uid", "attribute", "scraped_value", "flag",
                    "corroboration_0_4", "our_pred", "their_pred", "llm_pred",
                    "ocr_same_family_values", "ocr_salient", "image_link",
                    "claude_reading", "claude_confidence", "verified_value", "notes"])
        for i, r in enumerate(rows, 1):
            fmt = lambda p: f"{p[0]:g} {p[1]}" if p else "abstain"
            w.writerow([i, r.rec.sample_id, r.rec.entity_name, r.rec.entity_value,
                        r.flag, r.score, fmt(r.ours), fmt(r.theirs), fmt(r.llm),
                        "; ".join(f"{v:g} {u}" for v, u in r.ocr_same_family[:6]),
                        O.salient(O.clean(r.ocr_raw), 160).replace("\n", " | "),
                        r.rec.image_link, "", "", "", ""])
    print(f"\nwrote {args.out} ({len(rows)} rows, sorted by review priority)")


if __name__ == "__main__":
    main()
