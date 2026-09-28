"""Does the package image actually carry the answer?

The whole S3/S4/S5 arm is justified by the 16.7% of rows whose label is not in
the text.  That band is a mix of genuinely image-only cases and label noise, and
the ratio is unknown.  This probe measures it on a stratified sample before we
commit to an OCR pass over 145k images.

`literal_name` is the control: OCR must do well there, or the pipeline is broken
rather than the hypothesis.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev
import label_spans as LS
import ocr as O
import parse

N_TARGET = 300
N_CONTROL = 150
CACHE = "cache/ocr_probe.jsonl"


def main() -> None:
    recs = list(parse.load("data/train.csv"))
    pools: dict[str, list] = {"text_unrecoverable": [], "literal_name": []}
    for r in recs:
        gold = ev.gold_of(r)
        if not gold:
            continue
        b = ev.bucket(r, gold)
        if b in pools and O.image_path(r.image_file):
            pools[b].append((r, gold))

    random.seed(7)
    sample = ([("text_unrecoverable", *x) for x in random.sample(pools["text_unrecoverable"], N_TARGET)]
              + [("literal_name", *x) for x in random.sample(pools["literal_name"], N_CONTROL)])
    print(f"pools: unrecoverable {len(pools['text_unrecoverable']):,}, "
          f"literal {len(pools['literal_name']):,}  -> probing {len(sample)}\n", flush=True)

    stats: dict[str, Counter] = {b: Counter() for b in pools}
    rows_out, t0 = [], time.time()

    for i, (bucket, rec, gold) in enumerate(sample):
        gv, gu = gold
        raw = O.ocr_text(rec.image_file)
        text_ocr = O.clean(raw)
        title = rec.model_input(include_description=False)[:600]

        s = stats[bucket]
        s["n"] += 1
        s["ocr_empty"] += not raw.strip()
        s["ocr_raw_ok"] += LS.find_assignment(raw, gv, gu) is not None
        s["ocr_only"] += LS.find_assignment(text_ocr, gv, gu) is not None
        s["title_only"] += LS.find_assignment(title, gv, gu) is not None
        s["combined"] += LS.find_assignment(title + "\n" + text_ocr, gv, gu) is not None

        rows_out.append({"image": rec.image_file, "sample_id": rec.sample_id,
                         "bucket": bucket, "text": raw})
        if (i + 1) % 75 == 0:
            print(f"  {i+1}/{len(sample)}  ({(time.time()-t0)/(i+1):.2f}s/img)", flush=True)

    os.makedirs("cache", exist_ok=True)
    with open(CACHE, "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    print(f"\nOCR over {len(sample)} images in {(time.time()-t0)/60:.1f} min "
          f"({(time.time()-t0)/len(sample):.2f}s/image)\n")
    hdr = f"{'bucket':<22}{'n':>5}{'empty':>8}{'title':>8}{'ocr':>8}{'ocr+fix':>9}{'title+ocr':>11}"
    print(hdr); print("-" * len(hdr))
    for b, s in stats.items():
        n = s["n"] or 1
        print(f"{b:<22}{s['n']:>5}{100*s['ocr_empty']/n:>7.1f}%{100*s['title_only']/n:>7.1f}%"
              f"{100*s['ocr_raw_ok']/n:>7.1f}%{100*s['ocr_only']/n:>8.1f}%{100*s['combined']/n:>10.1f}%")
    print("\n'ocr' = raw Vision output, 'ocr+fix' = after digit/letter repair.")
    print(f"cached -> {CACHE}")


if __name__ == "__main__":
    main()
