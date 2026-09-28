"""Does the demo pipeline reproduce the evaluated system?

Runs `Pipeline` on the 406 validation rows of S3/S5, on the same cached OCR, and
compares row by row against what the Kaggle run (CUDA) and route_2024.py produced.
The VLM runs on MPS here, so a near-tied letter may flip; the check reports how
many rather than assuming none.

    python src/demo/parity.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.pipeline import Pipeline
import ocr as O

key = lambda f, a: f + "|" + a
vlm = {key(r["image_file"], r["attribute"]): r for r in json.load(open("kaggle/vlm_lora_results.json"))}
routed = json.load(open("kaggle/route_2024_rows.json"))
cache = O.load_cache("cache/ocr_ws2024.jsonl")

p = Pipeline()
n = cas_agree = vlm_agree = vlm_asked = r1_ok = r1_ref = 0
flips, t0 = [], time.time()
for k, r in vlm.items():
    gold = (float(r["gold_lo"]), float(r["gold_hi"]), r["gold_unit"])
    evs = {e["stage"]: e for e in p.run(r["attribute"], image_path=O.image_path(r["image_file"]),
                                          ocr_text=cache[r["image_file"]], gold=gold)}
    fin, v = evs["final"], evs["vlm"]
    n += 1
    # cascade correctness must match route_2024.py exactly - it is deterministic
    from rules import Prediction
    c = evs["cascade"]["cascade"]
    import evaluate as ev
    c_ok = bool(c) and bool(ev.score(Prediction(c["value"], c["unit"], ""), gold)["dim"])
    cas_agree += c_ok == routed[k]["cascade_ok"]
    if "pick" in v:
        vlm_asked += 1
        same = v["pred"] == r["pred"]
        vlm_agree += same
        if not same:
            flips.append(dict(row=k, kaggle=r["raw"], local=v["pick"], probs=v["probs"], kaggle_ok=r["dim"],
                              local_ok=fin["correct"]))
    r1_ok += fin["correct"]
    r1_ref += routed[k]["r1_ok"]
    if n % 50 == 0:
        print(f"  {n}/{len(vlm)}  {time.time() - t0:.0f}s", flush=True)

print(f"\nrows                         {n}")
print(f"cascade correctness agrees   {cas_agree}/{n}")
print(f"VLM answer agrees (2+ cands) {vlm_agree}/{vlm_asked}")
print(f"router dim: demo {r1_ok}/{n} = {100 * r1_ok / n:.1f}%   reported {r1_ref}/{n} = {100 * r1_ref / n:.1f}%")
for f in flips:
    print("  flip:", f)
json.dump(dict(n=n, cascade_agree=cas_agree, vlm_agree=vlm_agree, vlm_asked=vlm_asked,
               demo_ok=r1_ok, reported_ok=r1_ref, flips=flips), open("cache/demo_parity.json", "w"), indent=1)
