"""Corpus-wide audit of scraped labels - no human required.

Reading 175 images one at a time verifies 175 labels. Detecting the *scraper's
own systematic mistakes* verifies patterns across all 3,789 - and those
mistakes are legible: the 2024 labels were themselves extracted automatically,
so they inherit unit-confusion errors that leave a signature.

  "HOLD UP TO 52 CUPCAKES"          -> 52.0 ton      (count read as tons)
  "Support up to FULL HD 1080P"     -> 1080.0 hp     (resolution read as horsepower)
  "Add 4g to smoothies"             -> 4.0 gallon    (grams read as gallons)
  "63.5 to 83 cm"                   -> 63.5 ton      (centimetres read as tons)

Each rule below was confirmed by opening the image. This yields a *floor* on
the noise rate, never a ceiling: it only catches patterns we thought to look
for.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import ocr as O

VIDEO_RES = {480, 576, 720, 1080, 1440, 2160, 4320}

# Each rule: (name, predicate(attribute, low, high, unit, ocr_text), confirmed_by)
RULES = [
    ("resolution_as_horsepower",
     lambda a, lo, hi, u, t: a == "wattage" and u == "horsepower" and lo in VIDEO_RES,
     "image 12435: HDMI adapter, 'FULL HD 1080P' -> 1080.0 horsepower"),
    ("ton_on_consumer_good",
     lambda a, lo, hi, u, t: a == "maximum_weight_recommendation" and u == "ton",
     "image 43690: acrylic cupcake stand, 'HOLD UP TO 52 CUPCAKES' -> 52.0 ton"),
    ("gram_as_gallon",
     lambda a, lo, hi, u, t: (a == "item_volume" and u == "gallon"
                              and f"{lo:g}g" in t.replace(" ", "")),
     "'Add 4g to smoothies' -> 4.0 gallon"),
    ("percent_as_volume",
     lambda a, lo, hi, u, t: (a == "item_volume" and u in ("deciliter", "centiliter")
                              and f"{lo:g}%" in t),
     "'100% NATURAL' -> 100.0 decilitre"),
    ("screen_inches_as_volt",
     lambda a, lo, hi, u, t: (a == "voltage" and 15 <= lo <= 100 and lo == int(lo)
                              and f'{int(lo)}"' in t),
     "'65\"' on a television -> 65.0 volt"),
    ("microgram_weight_limit",
     lambda a, lo, hi, u, t: a == "maximum_weight_recommendation" and u == "microgram",
     "a load limit expressed in micrograms is not physical"),
]


def audit(recs, cache) -> tuple[list, Counter]:
    flagged, hits = [], Counter()
    for r in recs:
        lo, hi, u = r.gold
        t = O.clean(cache.get(r.image_file, ""))
        fired = [name for name, pred, _ in RULES if pred(r.entity_name, lo, hi, u, t)]
        if fired:
            hits.update(fired)
            flagged.append((r, fired))
    return flagged, hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.expanduser(
        "~/Multimodal_Product_Attribution_Extraction_SHARE/data/raw/working_set.csv"))
    ap.add_argument("--ocr", default="cache/ocr_2024.jsonl")
    ap.add_argument("--out", default="gold/provably_wrong.csv")
    ap.add_argument("--gold", action="store_true",
                    help="audit the 175-row held-out gold sheet instead")
    args = ap.parse_args()

    if args.gold:
        # The gold sheet is disjoint from the working set and uses its own
        # column names, so it must be audited separately - and it is stratified
        # 35-per-attribute, which over-represents the noisiest attributes.
        gold_csv = os.path.expanduser(
            "~/Multimodal_Product_Attribution_Extraction_SHARE/data/gold_test_set/review_sheet.csv")
        recs = []
        with open(gold_csv) as fh:
            for r in csv.DictReader(fh):
                rec = A.Record2024(sample_id=r["row_uid"], image_link=r["image_link"],
                                   entity_name=r["entity_name"],
                                   entity_value=r["scraped_entity_value"], group_id="")
                if rec.gold:
                    recs.append(rec)
        args.ocr = "cache/ocr_2024_gold.jsonl"
        args.out = "gold/provably_wrong_gold.csv"
    else:
        recs = [r for r in A.load(args.csv) if r.gold]
    cache = O.load_cache(args.ocr)
    flagged, hits = audit(recs, cache)
    n = len(recs)

    print(f"audited {n:,} labelled rows\n")
    hdr = f"{'rule':<28}{'n':>6}{'share':>8}   confirmed by"
    print(hdr); print("-" * 100)
    for name, _, why in RULES:
        c = hits.get(name, 0)
        print(f"{name:<28}{c:>6}{100*c/n:>7.2f}%   {why}")
    print("-" * 100)
    print(f"{'UNIQUE ROWS FLAGGED':<28}{len(flagged):>6}{100*len(flagged)/n:>7.2f}%")
    print(f"\nThis is a FLOOR on the label-noise rate, not an estimate of it:")
    print(f"it catches only the patterns these six rules describe.")

    per = Counter(r.entity_name for r, _ in flagged)
    print(f"\nby attribute:")
    tot = Counter(r.entity_name for r in recs)
    for a in sorted(tot, key=lambda x: -per.get(x, 0)):
        print(f"  {a:<32}{per.get(a,0):>5}/{tot[a]:<6}{100*per.get(a,0)/tot[a]:>6.2f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_uid", "attribute", "scraped_value", "rules_fired",
                    "ocr_evidence", "image_link"])
        for r, fired in flagged:
            w.writerow([r.sample_id, r.entity_name, r.entity_value, "|".join(fired),
                        O.salient(O.clean(cache.get(r.image_file, "")), 120).replace("\n", " | "),
                        r.image_link])
    print(f"\nwrote {args.out} ({len(flagged)} rows)")


if __name__ == "__main__":
    main()
