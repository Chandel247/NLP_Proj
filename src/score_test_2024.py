"""S6: the headline numbers on the untouched test split, scored once.

Reads the Kaggle run's per-row answers (kaggle/test/vlm_test_results.json), runs
the text cascade (S2) locally on the same rows, and applies the S5 router exactly
as it was fixed on validation: the VLM's answer whenever it gives one, the
cascade's otherwise. Nothing is chosen here - every rule was settled before this
file existed.

    python src/score_test_2024.py
"""
import json
import math
import sys
from collections import Counter
from math import comb

sys.path.insert(0, "src")
import torch

import amazon2024 as A
import evaluate as ev
import label_spans
import ocr as O
import rules2024
import tagger
import train_2024 as T

tagger.MAX_LEN = 224
tagger.set_unit_classes(tagger.UNIT_CLASSES_2024)
key = lambda f, a: f + "|" + a

vlm = {key(r["image_file"], r["attribute"]): r
       for r in json.load(open("kaggle/test/vlm_test_results.json"))}
LENGTH = {"depth", "width", "height"}
# Every gold test row: length rows are the ones the VLM answered (one leaked row
# was excluded when the package was built), the other five attributes go to the
# cascade alone, so the blended figure covers the whole task.
recs = [r for r in A.load_working_set(T.WS, split="test")
        if r.gold and (r.entity_name not in LENGTH or key(r.image_file, r.entity_name) in vlm)]
assert sum(r.entity_name in LENGTH for r in recs) == len(vlm)

dev = T.device()
tok = tagger.get_tokenizer()
model = tagger.SpanUnitTagger().to(dev)
model.load_state_dict(torch.load(f"{T.CKPT}/model.pt", map_location=dev))
model.eval()
cache = O.load_cache(T.OCR)

cas = {}
with torch.no_grad():
    for i in range(0, len(recs), 64):
        ch = recs[i:i + 64]
        texts = [label_spans.build_text(r, cache.get(r.image_file)) for r in ch]
        enc = tok(texts, truncation=True, max_length=tagger.MAX_LEN, padding="max_length",
                  return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(dev) for k, v in enc.items()}
        tg, un = model(enc["input_ids"], enc["attention_mask"])
        for r, text, t, o, u in zip(ch, texts, tg.argmax(-1).cpu().tolist(), offs.tolist(),
                                    un.argmax(-1).cpu().tolist()):
            pt = tagger.decode_prediction(text, t, o, u)
            ps = rules2024.predict(r.entity_name, O.clean(cache.get(r.image_file, "")))
            p = ps if pt.abstained else pt
            cas[key(r.image_file, r.entity_name)] = bool(ev.score(p, r.gold)["dim"])


def w(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100*k/n:5.1f}% [{100*(c-h):.1f}, {100*(c+h):.1f}]"


def mcn(a, b):
    n = a + b; k = min(a, b)
    return min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


K = list(vlm)
n = len(K)
V = {k: bool(vlm[k]["dim"]) for k in K}
C = {k: cas[k] for k in K}
R = {k: (V[k] if vlm[k]["pred"] is not None else C[k]) for k in K}
O_ = {k: V[k] or C[k] for k in K}

print(f"== TEST split, {n} depth/width/height rows (validation figure in brackets) ==")
for name, S, val in (("cascade (S2)", C, "32.8"), ("VLM (S3)", V, "51.5"),
                     ("router (S5)", R, "54.9"), ("oracle", O_, "59.9")):
    print(f"  {name:13s} {w(sum(S.values()), n)}   (val {val}%)")
for a_, b_, label in ((V, C, "VLM vs cascade"), (R, V, "router vs VLM")):
    g = sum(a_[k] and not b_[k] for k in K); l = sum(b_[k] and not a_[k] for k in K)
    print(f"  {label}: +{g} / -{l}  McNemar p={mcn(g, l):.1e}")

print("\n== by attribute ==")
for att in ("depth", "width", "height"):
    ks = [k for k in K if vlm[k]["attribute"] == att]
    print(f"  {att:7s} n={len(ks):4d}  cascade {w(sum(C[k] for k in ks), len(ks))}  "
          f"VLM {w(sum(V[k] for k in ks), len(ks))}  router {w(sum(R[k] for k in ks), len(ks))}")

gp = [k for k in K if vlm[k]["n_cands"] >= 2 and vlm[k]["gold_index"] not in (-1, "-1")]
print(f"\n2+ candidates, gold present: VLM {w(sum(V[k] for k in gp), len(gp))}  (val 70.8%)")
print(f"gold among candidates (selector ceiling): "
      f"{100*sum(vlm[k]['gold_index'] not in (-1, '-1') for k in K)/n:.1f}%  (val 70.0%)")
print("n_cands:", dict(sorted(Counter(min(vlm[k]['n_cands'], 2) for k in K).items())), "(2 = 2+)")

other = [k for k in cas if k not in vlm]
atts = Counter(k.split("|")[1] for k in other)
print(f"\n== whole task, all {len(cas)} test rows ==")
for att in sorted(atts, key=lambda a: -atts[a]):
    ks = [k for k in other if k.endswith("|" + att)]
    print(f"  {att:30s} n={len(ks):4d}  cascade {w(sum(cas[k] for k in ks), len(ks))}")
ko = sum(cas[k] for k in other)
print(f"  {'other five, cascade':30s} n={len(other):4d}  {w(ko, len(other))}")
print(f"  {'ALL, cascade only':30s} n={len(cas):4d}  {w(ko + sum(C.values()), len(cas))}")
print(f"  {'ALL, cascade + router':30s} n={len(cas):4d}  {w(ko + sum(R.values()), len(cas))}")

json.dump({k: dict(cascade_ok=C[k], vlm_ok=V[k], router_ok=R[k], n_cands=vlm[k]["n_cands"])
           for k in K}, open("kaggle/test/score_test_rows.json", "w"), indent=1)
