"""Choose the demo's example rows and record a full run of each for replay mode.

Every example comes from the test split, which no decision was tuned on, and the
set is chosen to show each path through the system - including an honest failure.
For length rows the blank-image control is recorded too.

    python src/demo/make_examples.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.pipeline import LENGTH, Pipeline
import amazon2024 as A
import ocr as O

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.json")
rng = random.Random(7)

test = {(r.image_file, r.entity_name): r for r in A.load_working_set("data2024/working_set.csv", split="test")
        if r.gold and O.image_path(r.image_file)}
scored = json.load(open("kaggle/test/score_test_rows.json"))
vlm = {r["image_file"] + "|" + r["attribute"]: r for r in json.load(open("kaggle/test/vlm_test_results.json"))}
p = Pipeline()


def run(rec, control="real"):
    return list(p.run(rec.entity_name, image_path=O.image_path(rec.image_file), control=control, gold=rec.gold))


def pick(keys, want, k, title, note):
    """First k rows (shuffled) whose live real-image run satisfies `want`."""
    keys = list(keys); rng.shuffle(keys)
    got = []
    for key in keys:
        f, a = key.split("|")
        rec = test.get((f, a))
        if rec is None:
            continue
        real = run(rec)
        blank = run(rec, "blank") if a in LENGTH else None
        if want(real, blank):
            got.append(dict(id=f"{a}-{f.split('.')[0]}", image_file=f, attribute=a, gold=list(rec.gold),
                            title=title, note=note, events={"real": real, **({"blank": blank} if blank else {})}))
            print(f"  {title}: {key}", flush=True)
            if len(got) == k:
                break
    return got


fin = lambda evs: evs[-1]
vlm_ev = lambda evs: next(e for e in evs if e["stage"] == "vlm")
examples = []

# 1. The headline: several same-family numbers, the cascade picks wrong, the VLM right,
#    and with a blank page it goes wrong again - the model is reading the photo.
for attr in ("width", "height"):
    keys = [k for k, s in scored.items() if k.endswith("|" + attr) and s["n_cands"] >= 3
            and not s["cascade_ok"] and s["vlm_ok"]]
    examples += pick(keys, lambda r, b: fin(r)["correct"] and max(vlm_ev(r).get("probs") or [0]) >= 0.6
                     and not fin(b)["correct"], 1,
                     f"{attr}: the photo decides",
                     "OCR reads several lengths and nothing in the text says which is the "
                     f"{attr}. The cascade guesses wrong; the fine-tuned VLM picks right - and "
                     "loses it when the photo is replaced by a blank page.")

# 2. Depth, where the text alone often suffices (the S4 control's +9 point finding).
keys = [k for k, s in scored.items() if k.endswith("|depth") and s["n_cands"] >= 2 and s["vlm_ok"]]
examples += pick(keys, lambda r, b: fin(r)["correct"], 1, "depth: a size chart",
                 "Depth is often the largest number on a size chart, so text carries more of it.")

# 3. No length candidate at all: the router falls back to the cascade.
keys = [k for k, s in scored.items() if s["n_cands"] == 0 and s["cascade_ok"]]
examples += pick(keys, lambda r, b: fin(r)["correct"] and fin(r)["source"] == "cascade", 1,
                 "fallback: nothing for the VLM",
                 "OCR produced no clean length candidate, so the VLM has nothing to choose from; "
                 "the tagger's span extraction answers instead.")

# 4. The five non-length attributes: the dimensional filter plus the tagger.
nonlen = [f"{f}|{a}" for (f, a) in test if a not in LENGTH]
for attr, title, family in (("item_weight", "item weight", "mass"), ("wattage", "wattage", "power"),
                            ("voltage", "voltage", "voltage")):
    examples += pick([k for k in nonlen if k.endswith("|" + attr)],
                     lambda r, b: fin(r)["correct"] and len(next(e for e in r if e["stage"] == "candidates")["items"]) >= 2,
                     1, f"{title}: unit family filters the noise",
                     f"Only a {family} quantity can answer this, which discards barcodes and batch "
                     "numbers for free.")

# 5. An honest failure: the right value is not among what OCR read.
keys = [k for k, v in vlm.items() if v["n_cands"] >= 2 and v["gold_index"] < 0]
examples += pick(keys, lambda r, b: not fin(r)["correct"], 1, "failure: OCR missed it",
                 "The correct value is not among the numbers OCR read, so a selector cannot reach it. "
                 "29% of test length rows look like this - OCR recall is the next ceiling.")

json.dump(examples, open(OUT, "w"), indent=1)
print(f"wrote {len(examples)} examples -> {OUT}")
