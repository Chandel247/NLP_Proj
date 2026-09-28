"""S5: route between the text cascade (S2) and the fine-tuned VLM (S3) on length rows.

The routing rule is fixed in advance, not fitted: take the VLM's answer whenever it
gives one, and fall back to the cascade when OCR produced no length candidate for it
to choose from. Its evaluation rows are the same 406 held-out rows as S3, so fitting
anything here would be tuning on the test set - the diagnostics printed after the
headline are for deciding whether a learned router is worth building, not for
picking a rule.

    python src/route_2024.py
"""
import io, json, math, sys, csv, tarfile
from collections import Counter
from math import comb
sys.path.insert(0, "src")
import torch
import train_2024 as T
import amazon2024 as A, ocr as O, tagger, label_spans, rules2024, evaluate as ev, normalize as N

tagger.MAX_LEN = 224
tagger.set_unit_classes(tagger.UNIT_CLASSES_2024)
key = lambda f, a: f + "|" + a
vlm = {key(r["image_file"], r["attribute"]): r for r in json.load(open("kaggle/vlm_lora_results.json"))}
with tarfile.open("kaggle/vlm_lora_data.tar.gz") as t:
    _eval = t.extractfile("vlm_lora/eval.csv").read().decode()
cands = {key(r["image_file"], r["attribute"]): r for r in csv.DictReader(io.StringIO(_eval))}

recs = [r for r in A.load_working_set(T.WS, split="val") if r.gold and key(r.image_file, r.entity_name) in vlm]
dev = T.device(); tok = tagger.get_tokenizer()
model = tagger.SpanUnitTagger().to(dev)
model.load_state_dict(torch.load(f"{T.CKPT}/model.pt", map_location=dev)); model.eval()
cache = O.load_cache(T.OCR)
cas = {}
with torch.no_grad():
    for i in range(0, len(recs), 64):
        ch = recs[i:i+64]
        texts = [label_spans.build_text(r, cache.get(r.image_file)) for r in ch]
        enc = tok(texts, truncation=True, max_length=tagger.MAX_LEN, padding="max_length", return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping"); enc = {k: v.to(dev) for k, v in enc.items()}
        tg, un = model(enc["input_ids"], enc["attention_mask"])
        for r, text, t, o, u in zip(ch, texts, tg.argmax(-1).cpu().tolist(), offs.tolist(), un.argmax(-1).cpu().tolist()):
            pt = tagger.decode_prediction(text, t, o, u)
            ps = rules2024.predict(r.entity_name, O.clean(cache.get(r.image_file, "")))
            p = ps if pt.abstained else pt
            cas[key(r.image_file, r.entity_name)] = dict(rec=r, pred=p, ok=bool(ev.score(p, r.gold)["dim"]),
                                                      source=("rules:" if pt.abstained else "tagger:") + p.rule)
assert len(cas) == 406

def w(k, n, z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return f"{100*k/n:5.1f}% [{100*(c-h):.1f}, {100*(c+h):.1f}]"
def mcn(a, b):
    n=a+b; k=min(a,b); return min(1, 2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.0

K = list(vlm)
V = {k: bool(vlm[k]["dim"]) for k in K}
C = {k: cas[k]["ok"] for k in K}
n = len(K)
# --- pre-specified router: VLM when it answers, cascade otherwise ---
R1 = {k: (V[k] if vlm[k]["pred"] is not None else C[k]) for k in K}
print("== pre-specified router (fixed before scoring): VLM if it answers, else cascade ==")
for name, S in (("cascade (S2)", C), ("VLM (S3)", V), ("router R1", R1)):
    print(f"  {name:14s} {w(sum(S.values()), n)}")
orc = {k: V[k] or C[k] for k in K}
print(f"  {'oracle':14s} {w(sum(orc.values()), n)}")
g = sum(R1[k] and not V[k] for k in K); l = sum(V[k] and not R1[k] for k in K)
print(f"  R1 vs VLM: +{g} / -{l}  McNemar p={mcn(g,l):.1e}")
vabst = [k for k in K if vlm[k]["pred"] is None]
print(f"  VLM gave no answer on {len(vabst)} rows (n_cands: {dict(Counter(vlm[k]['n_cands'] for k in vabst))}); cascade right on {sum(C[k] for k in vabst)} of them")

print("\n== the residual: rows where the cascade is right and R1 is wrong ==")
res = [k for k in K if C[k] and not R1[k]]
print(f"  {len(res)} rows = {100*len(res)/n:.1f} pts of oracle headroom left")
def in_cands(k):
    p = cas[k]["pred"]; cs = cands[k]["candidates"]
    if p.abstained: return False
    b = N.to_base(p.value, p.unit)
    for part in cs.split(";"):
        bits = part.split()
        if len(bits) == 2 and N.normalize_unit(bits[1]):
            if abs(N.to_base(float(bits[0]), N.normalize_unit(bits[1])) - b) <= 0.01*max(b,1e-9): return True
    return False
print("  by n_cands:", dict(Counter(vlm[k]["n_cands"] for k in res)))
print("  cascade source:", dict(Counter(cas[k]["source"] for k in res)))
print("  cascade answer is one of the VLM's candidates:", sum(in_cands(k) for k in res), "/", len(res))

print("\n== what could a router see? (diagnostic only - NOT for choosing a rule on these rows) ==")
both = [k for k in K if vlm[k]["pred"] is not None and not cas[k]["pred"].abstained]
agree = [k for k in both if abs(N.to_base(*vlm[k]["pred"]) - N.to_base(cas[k]["pred"].value, cas[k]["pred"].unit)) <= 0.01*max(N.to_base(*vlm[k]["pred"]),1e-9)]
dis = [k for k in both if k not in agree]
print(f"  both answered {len(both)} | agree {len(agree)} (VLM right {w(sum(V[k] for k in agree),len(agree)) if agree else '-'}) | disagree {len(dis)}")
print(f"  on disagreements: VLM right {sum(V[k] for k in dis)}, cascade right {sum(C[k] for k in dis)}, neither {sum(not V[k] and not C[k] for k in dis)}")
out_c = [k for k in dis if not in_cands(k)]
print(f"  disagreements where cascade's answer is NOT a candidate: {len(out_c)}; VLM right {sum(V[k] for k in out_c)}, cascade right {sum(C[k] for k in out_c)}")
json.dump({k: dict(cascade_ok=C[k], vlm_ok=V[k], r1_ok=R1[k], n_cands=vlm[k]["n_cands"], cascade_source=cas[k]["source"],
                   cascade_in_cands=in_cands(k), vlm_answered=vlm[k]["pred"] is not None) for k in K},
          open("kaggle/route_2024_rows.json", "w"), indent=1)
