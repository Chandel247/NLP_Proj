"""Compare the corpus before and after OCR.

Three questions:
  1. Does the full training split reproduce the 300-image probe's decomposition
     of the text-unrecoverable band, or was the probe lucky?
  2. How much does OCR raise weak-supervision coverage - i.e. how much more
     training data does S3 get than S1 had?
  3. What is the realised accuracy headroom, per difficulty bucket?
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev
import label_spans as LS
import normalize as N
import ocr as O
import parse

OCR_CHARS = 800          # what S3 would actually feed the model
NUM = re.compile(r"\d+(?:\.\d+)?")

# The probe's numbers, for the side-by-side.
PROBE = {"solved": 27.7, "no_ocr_text": 9.7,
         "gold_number_present_no_program": 7.7, "gold_number_absent": 55.0}


def main() -> None:
    cache = O.load_cache("cache/ocr_train.jsonl")
    print(f"OCR cache: {len(cache):,} images\n")

    before = Counter()
    after = Counter()
    totals = Counter()
    band = Counter()
    missing_img = 0

    for r in parse.load("data/train.csv"):
        gold = ev.gold_of(r)
        if gold is None:
            continue
        gv, gu = gold
        b = ev.bucket(r, gold)
        totals[b] += 1

        title = r.model_input(include_description=False)[:600]
        raw = cache.get(r.image_file)
        if raw is None:
            missing_img += 1
            continue
        text = O.clean(raw)[:OCR_CHARS]

        a_before = LS.find_assignment(title, gv, gu) is not None
        a_after = a_before or LS.find_assignment(title + "\n" + text, gv, gu) is not None
        before[b] += a_before
        after[b] += a_after

        if b == "text_unrecoverable":
            if a_after:
                band["solved"] += 1
            elif not raw.strip():
                band["no_ocr_text"] += 1
            else:
                nums = [float(x) for x in NUM.findall(text)]
                hit = any(x and abs(x - gv) / gv <= 0.01 for x in nums)
                for q in N.find_quantities(text):
                    cv, _ = N.catalog_convert(q.value, q.unit, gu)
                    if cv and abs(cv - gv) / gv <= 0.01:
                        hit = True
                band["gold_number_present_no_program" if hit else "gold_number_absent"] += 1

    n = sum(totals.values())
    if missing_img:
        print(f"note: {missing_img:,} rows had no cached OCR (image absent)\n")

    print("weak-supervision coverage, before vs after OCR")
    hdr = f"{'bucket':<22}{'rows':>9}{'before':>9}{'after':>9}{'gain':>8}"
    print(hdr); print("-" * len(hdr))
    for b in ev.BUCKETS:
        t = totals[b]
        if not t:
            continue
        pb, pa = 100 * before[b] / t, 100 * after[b] / t
        print(f"{b:<22}{t:>9,}{pb:>8.1f}%{pa:>8.1f}%{pa-pb:>+7.1f}")
    tb, ta = 100 * sum(before.values()) / n, 100 * sum(after.values()) / n
    print(f"{'ALL':<22}{n:>9,}{tb:>8.1f}%{ta:>8.1f}%{ta-tb:>+7.1f}")

    bn = sum(band.values())
    print(f"\ntext-unrecoverable band: probe (n=300) vs full split (n={bn:,})")
    hdr = f"{'outcome':<38}{'probe':>9}{'full':>9}{'delta':>8}"
    print(hdr); print("-" * len(hdr))
    for k in ("solved", "no_ocr_text", "gold_number_present_no_program", "gold_number_absent"):
        full = 100 * band[k] / bn
        print(f"{k:<38}{PROBE[k]:>8.1f}%{full:>8.1f}%{full-PROBE[k]:>+7.1f}")

    noise = band["gold_number_absent"] / bn * totals["text_unrecoverable"] / n
    print(f"\nlabel-noise floor: {100*noise:.1f}% of the corpus "
          f"({band['gold_number_absent']:,} rows) carry a value found neither "
          f"in the text nor on the package")
    print(f"OCR headroom:      {ta-tb:+.1f} points of solvable rows")


if __name__ == "__main__":
    main()
