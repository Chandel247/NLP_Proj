"""Head-to-head against the teammate pipeline on their 175-row gold set (five non-length attributes).

    python src/compare_teammate.py
"""
import sys, os, csv, math
from collections import Counter
from math import comb
sys.path.insert(0, "src")
import torch
import verify_labels as V, tagger, label_spans, rules2024, ocr as O, train_2024 as T, amazon2024 as A

tagger.MAX_LEN = 224; tagger.set_unit_classes(tagger.UNIT_CLASSES_2024)
rows = V.build_rows()
cache = O.load_cache("cache/ocr_2024_gold.jsonl")
dev = T.device(); tok = tagger.get_tokenizer()
m = tagger.SpanUnitTagger().to(dev); m.load_state_dict(torch.load(f"{T.CKPT}/model.pt", map_location=dev)); m.eval()
casc = {}
with torch.no_grad():
    for i in range(0, len(rows), 64):
        ch = rows[i:i+64]
        texts = [label_spans.build_text(r.rec, cache.get(r.rec.image_file)) for r in ch]
        enc = tok(texts, truncation=True, max_length=tagger.MAX_LEN, padding="max_length", return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping"); enc = {k: v.to(dev) for k, v in enc.items()}
        tg, un = m(enc["input_ids"], enc["attention_mask"])
        for r, t, tt, o, u in zip(ch, texts, tg.argmax(-1).cpu().tolist(), offs.tolist(), un.argmax(-1).cpu().tolist()):
            pt = tagger.decode_prediction(t, tt, o, u)
            p = rules2024.predict(r.rec.entity_name, O.clean(cache.get(r.rec.image_file, ""))) if pt.abstained else pt
            casc[r.rec.sample_id] = None if p.abstained else (p.value, p.unit)

# leakage: gold photos vs our working-set train split, by CDN basename
train = {r.image_file for r in A.load_working_set(T.WS, split="train")}
allws = {r.image_file: None for r in A.load_working_set(T.WS)}
base = lambda r: os.path.basename(r.rec.image_link)
leak_train = [r for r in rows if base(r) in train]
print(f"gold rows {len(rows)} | photo in our train split: {len(leak_train)} | in working set at all: {sum(base(r) in allws for r in rows)}")

wrong = {r["row_uid"] for r in csv.DictReader(open("gold/provably_wrong_gold.csv"))} if os.path.exists("gold/provably_wrong_gold.csv") else set()

def wil(k, n, z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f"[{100*(c-h):4.1f}, {100*(c+h):4.1f}]"
def mcn(a,b):
    n=a+b; k=min(a,b); return min(1, 2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0

systems = {"teammate: EasyOCR+DistilBERT": lambda r: r.theirs,
           "teammate: gpt-4o-mini (API)": lambda r: r.llm,
           "ours: S0 rules": lambda r: r.ours,
           "ours: tagger+rules cascade": lambda r: casc[r.rec.sample_id]}
for label, sel in (("all gold rows", lambda r: True),
                   ("excluding provably-wrong labels", lambda r: r.rec.sample_id not in wrong),
                   ("excluding provably-wrong AND photos in our train split", lambda r: r.rec.sample_id not in wrong and base(r) not in train)):
    ks = [r for r in rows if sel(r)]; n = len(ks)
    print(f"\n== {label}: n={n} ==")
    print(f"{'system':32s}{'acc':>7s}{'95% CI':>15s}{'cover':>8s}{'prec|ans':>10s}")
    for name, g in systems.items():
        ok = sum(V.agrees(g(r), r.rec.gold) for r in ks); ans = sum(g(r) is not None for r in ks)
        print(f"{name:32s}{100*ok/n:6.1f}%{wil(ok,n):>15s}{100*ans/n:7.1f}%{(100*ok/ans if ans else 0):9.1f}%")
    a = sum(V.agrees(casc[r.rec.sample_id], r.rec.gold) and not V.agrees(r.theirs, r.rec.gold) for r in ks)
    b = sum(V.agrees(r.theirs, r.rec.gold) and not V.agrees(casc[r.rec.sample_id], r.rec.gold) for r in ks)
    print(f"  cascade vs teammate pipeline: +{a} / -{b}  McNemar p={mcn(a,b):.1e}")

print("\n== per attribute, all rows (acc) ==")
for att in sorted({r.rec.entity_name for r in rows}):
    ks = [r for r in rows if r.rec.entity_name == att]; n=len(ks)
    print(f"  {att:30s} n={n:3d}  " + "  ".join(f"{nm.split(': ')[1][:12]:>12s} {100*sum(V.agrees(g(r), r.rec.gold) for r in ks)/n:5.1f}%" for nm, g in systems.items()))

# where the teammate commits: precision head to head on rows both answer
both = [r for r in rows if r.theirs is not None and casc[r.rec.sample_id] is not None]
print(f"\nrows both commit: {len(both)} | teammate right {sum(V.agrees(r.theirs, r.rec.gold) for r in both)} | cascade right {sum(V.agrees(casc[r.rec.sample_id], r.rec.gold) for r in both)}")
# accuracy-coverage: cascade restricted to the rows the teammate answered
ta = [r for r in rows if r.theirs is not None]
print(f"on the {len(ta)} rows the teammate answers: teammate {sum(V.agrees(r.theirs, r.rec.gold) for r in ta)} right, cascade {sum(V.agrees(casc[r.rec.sample_id], r.rec.gold) for r in ta)} right")
