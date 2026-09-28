"""Train and evaluate the tagger on the rebuilt 2024 working set.

Reported per attribute and split by dimensional family, because the corpus is
two tasks: for the five non-length attributes S0 already captures 90% of what
OCR makes available, while for depth/width/height it captures 46%. The whole
question is whether a learned tagger closes that second gap.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import amazon2024 as A
import evaluate as ev
import label_spans
import ocr as O
import rules2024
import tagger

CKPT = "cache/t2024"
WS = "data2024/working_set.csv"
OCR = "cache/ocr_ws2024.jsonl"
LENGTH = {"depth", "width", "height"}


class SpanDataset(Dataset):
    def __init__(self, rows, tok):
        self.enc = tagger.encode(rows, tok)

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(args):
    dev = device()
    tok = tagger.get_tokenizer()
    rows = tagger.load_jsonl("cache/spans2024_train.jsonl", limit=args.limit)
    print(f"training on {len(rows):,} weakly-labelled examples  (device: {dev})", flush=True)

    dl = DataLoader(SpanDataset(rows, tok), batch_size=args.batch, shuffle=True, drop_last=True)
    model = tagger.SpanUnitTagger().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="linear")
    w = torch.ones(len(label_spans.TAGS), device=dev)
    w[1:] = args.tag_weight          # every B-/I- tag, not O
    tl = nn.CrossEntropyLoss(ignore_index=-100, weight=w)
    ul = nn.CrossEntropyLoss()

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train(); run = 0.0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            tg, un = model(b["input_ids"], b["attention_mask"])
            loss = (tl(tg.reshape(-1, tg.shape[-1]), b["labels"].reshape(-1))
                    + ul(un, b["unit_labels"]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); step += 1
            if step % 100 == 0:
                rate = step / (time.time() - t0)
                print(f"  epoch {epoch+1} step {step}/{total}  loss {run/100:.4f}  "
                      f"{rate:.1f} it/s  eta {(total-step)/rate/60:.1f} min", flush=True)
                run = 0.0
    os.makedirs(CKPT, exist_ok=True)
    torch.save(model.state_dict(), f"{CKPT}/model.pt")
    print(f"saved {CKPT}/model.pt after {step} steps in {(time.time()-t0)/60:.1f} min")


@torch.no_grad()
def run_eval(args):
    dev = device()
    tok = tagger.get_tokenizer()
    model = tagger.SpanUnitTagger().to(dev)
    model.load_state_dict(torch.load(f"{CKPT}/model.pt", map_location=dev))
    model.eval()

    cache = O.load_cache(OCR)
    recs = [r for r in A.load_working_set(WS, split=args.split) if r.gold]
    print(f"evaluating on {len(recs):,} {args.split} rows\n", flush=True)

    stats = {name: {} for name in ("S0", "tagger", "cascade")}
    fell = 0
    B = 64
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        texts = [label_spans.build_text(r, cache.get(r.image_file)) for r in chunk]
        enc = tok(texts, truncation=True, max_length=tagger.MAX_LEN, padding="max_length",
                  return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(dev) for k, v in enc.items()}
        tg, un = model(enc["input_ids"], enc["attention_mask"])
        tags, units = tg.argmax(-1).cpu().tolist(), un.argmax(-1).cpu().tolist()
        for r, text, t, o, u in zip(chunk, texts, tags, offs.tolist(), units):
            p_tag = tagger.decode_prediction(text, t, o, u)
            p_s0 = rules2024.predict(r.entity_name, O.clean(cache.get(r.image_file, "")))
            p_cas = p_s0 if p_tag.abstained else p_tag
            fell += p_tag.abstained
            for name, p in (("S0", p_s0), ("tagger", p_tag), ("cascade", p_cas)):
                c = stats[name].setdefault(r.entity_name, Counter())
                c["n"] += 1
                c["abstain"] += p.abstained
                for k, v in ev.score(p, r.gold).items():
                    c[k] += v

    def show(name):
        per = stats[name]
        n = sum(c["n"] for c in per.values())
        print(f"--- {name} ---")
        hdr = f"{'attribute':<32}{'n':>7}{'cover':>8}{'value':>8}{'dim':>8}"
        print(hdr); print("-" * len(hdr))
        for a in sorted(per, key=lambda x: -per[x]["n"]):
            c = per[a]
            print(f"{a:<32}{c['n']:>7,}{100*(c['n']-c['abstain'])/c['n']:>7.1f}%"
                  f"{100*c['value']/c['n']:>7.1f}%{100*c['dim']/c['n']:>7.1f}%")
        for label, sel in (("LENGTH", LENGTH), ("OTHER", set(per) - LENGTH)):
            sn = sum(per[a]["n"] for a in sel)
            sd = sum(per[a]["dim"] for a in sel)
            print(f"{label:<32}{sn:>7,}{'':>8}{'':>8}{100*sd/sn:>7.1f}%")
        alld = sum(c["dim"] for c in per.values())
        print(f"{'ALL':<32}{n:>7,}{'':>8}{'':>8}{100*alld/n:>7.1f}%\n")
        return 100 * alld / n

    res = {name: show(name) for name in ("S0", "tagger", "cascade")}
    print(f"tagger abstained on {fell:,} rows ({100*fell/sum(c['n'] for c in stats['S0'].values()):.1f}%)")
    print(f"\ndim accuracy:  S0 {res['S0']:.1f}%   tagger {res['tagger']:.1f}%   "
          f"cascade {res['cascade']:.1f}%   (cascade - S0 = {res['cascade']-res['S0']:+.1f} pts)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--tag-weight", type=float, default=8.0,
                    help="loss weight on span tags relative to O")
    ap.add_argument("--split", default="val")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    tagger.MAX_LEN = args.max_len
    tagger.set_unit_classes(tagger.UNIT_CLASSES_2024)
    if not args.eval_only:
        train(args)
    run_eval(args)


if __name__ == "__main__":
    main()
