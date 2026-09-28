"""Why does the model miss when the answer is right there in the title?

`literal_name` carries 12.7 of the ~25 recoverable points - half the remaining
loss, in the bucket where the gold value appears verbatim in the item name.
This classifies those failures so the next architectural change is chosen from
evidence rather than intuition.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev
import label_spans
import normalize as N
import ocr as O
import parse
import rules
import splits
import tagger

NUM = re.compile(r"\d+(?:\.\d+)?")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="literal_name")
    ap.add_argument("--ckpt", default="cache/s3")
    ap.add_argument("--max-len", type=int, default=288)
    args = ap.parse_args()
    tagger.MAX_LEN = args.max_len

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = tagger.get_tokenizer()
    model = tagger.SpanUnitTagger().to(dev)
    model.load_state_dict(torch.load(f"{args.ckpt}/model.pt", map_location=dev))
    model.eval()

    recs = list(parse.load("data/train.csv"))
    assign = splits.assign(recs)
    cache = O.load_cache("cache/ocr_train.jsonl")
    # Did the weak-supervision search find any program explaining this row's
    # label?  Val rows are never in the *train* span file, so comparing against
    # that would be trivially 100%; the meaningful question is whether the row
    # is the kind weak supervision can explain at all.
    solvable = set()
    with open("cache/spans_ocr_val.jsonl") as fh:
        for line in fh:
            solvable.add(line.split('"sample_id": "', 1)[1].split('"', 1)[0])

    val = [r for r in recs if assign[r.sample_id] == "val" and ev.gold_of(r)
           and ev.bucket(r, ev.gold_of(r)) == args.bucket]
    print(f"{args.bucket}: {len(val):,} val rows\n")

    kinds: Counter = Counter()
    unexplained: Counter = Counter()
    examples: list = []
    B = 64
    for i in range(0, len(val), B):
        chunk = val[i:i + B]
        texts = [label_spans.build_text(r, cache.get(r.image_file)) for r in chunk]
        enc = tok(texts, truncation=True, max_length=tagger.MAX_LEN,
                  padding="max_length", return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(dev) for k, v in enc.items()}
        tl, ul = model(enc["input_ids"], enc["attention_mask"])
        tags, units = tl.argmax(-1).cpu().tolist(), ul.argmax(-1).cpu().tolist()

        for r, text, t, o, u in zip(chunk, texts, tags, offs.tolist(), units):
            gv, gu = ev.gold_of(r)
            p = tagger.decode_prediction(text, t, o, u)
            if p.abstained:
                p = rules.predict(r)          # the cascade's behaviour
                if p.abstained:
                    kinds["both abstained"] += 1
                    unexplained["both abstained"] += r.sample_id not in solvable
                    continue
            if abs(p.value - gv) / gv <= 0.01:
                kinds["correct"] += 1
                continue

            ratio = p.value / gv if gv else 0
            near_int = min((abs(ratio - k) for k in range(1, 25)), default=9)
            near_inv = min((abs(1 / ratio - k) for k in range(1, 25)), default=9) if ratio else 9
            title_nums = [float(x) for x in NUM.findall(r.item_name)]
            if near_int < 0.02 or near_inv < 0.02:
                kind = "pack multiplier wrong"
            elif any(abs(p.value - x) / max(x, 1e-9) <= 0.01 for x in title_nums):
                kind = "picked another number from the title"
            elif N.family(p.unit) != N.family(gu):
                kind = "wrong dimensional family"
            else:
                kind = "other"
            kinds[kind] += 1
            unexplained[kind] += r.sample_id not in solvable
            if len(examples) < 8 and kind in ("picked another number from the title",
                                              "pack multiplier wrong"):
                examples.append((kind, gv, gu, p.value, p.unit, r.item_name[:70]))

    n = sum(kinds.values())
    print(f"{'outcome':<38}{'n':>7}{'share':>8}{'no program explains label':>27}")
    print("-" * 80)
    for k, c in kinds.most_common():
        frac = f"{100*unexplained[k]/c:.0f}%" if c and k != "correct" else "-"
        print(f"{k:<38}{c:>7,}{100*c/n:>7.1f}%{frac:>27}")
    print("\n--- examples ---")
    for kind, gv, gu, pv, pu, name in examples:
        print(f"  [{kind}] gold {gv} {gu} -> pred {pv:.4g} {pu}\n     {name}")


if __name__ == "__main__":
    main()
