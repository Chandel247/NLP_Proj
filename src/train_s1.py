"""Train and evaluate S1.

  python src/train_s1.py --epochs 2
  python src/train_s1.py --eval-only

Evaluation deliberately runs over *every* val row, not just the 68.8% that
carry weak span labels.  S0 is scored on all of them, so S1 must be too;
anything it cannot tag counts as an abstention against it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev
import label_spans
import ocr as O
import parse
import rules
import splits
import tagger

# The OCR variant needs a longer window: title alone exceeds 192 tokens on only
# 0.4% of rows, but title+OCR does on 15.4% (p99 = 286).  The extra capacity is
# therefore spent almost entirely on the package channel, not on more title,
# which keeps the S1 -> S3 comparison a test of OCR rather than of context size.
VARIANTS = {
    "text": {"prefix": "spans",     "ckpt": "cache/s1", "max_len": 192, "ocr": False},
    "ocr":  {"prefix": "spans_ocr", "ckpt": "cache/s3", "max_len": 288, "ocr": True},
}


class SpanDataset(Dataset):
    def __init__(self, rows, tok):
        self.enc = tagger.encode(rows, tok)

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(args) -> None:
    dev = device()
    tok = tagger.get_tokenizer()
    rows = tagger.load_jsonl(f"cache/{args.cfg['prefix']}_train.jsonl", limit=args.limit)
    print(f"training on {len(rows):,} weakly-labelled examples  (device: {dev})")

    dl = DataLoader(SpanDataset(rows, tok), batch_size=args.batch, shuffle=True, drop_last=True)
    model = tagger.SpanUnitTagger().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=0.1, anneal_strategy="linear")
    tag_loss = nn.CrossEntropyLoss(ignore_index=-100)
    unit_loss = nn.CrossEntropyLoss()

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            tag_logits, unit_logits = model(batch["input_ids"], batch["attention_mask"])
            loss = (tag_loss(tag_logits.reshape(-1, tag_logits.shape[-1]), batch["labels"].reshape(-1))
                    + unit_loss(unit_logits, batch["unit_labels"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run += loss.item(); step += 1
            if step % 100 == 0:
                rate = step / (time.time() - t0)
                eta = (total - step) / rate / 60
                print(f"  epoch {epoch+1} step {step}/{total}  loss {run/100:.4f}  "
                      f"{rate:.1f} it/s  eta {eta:.1f} min", flush=True)
                run = 0.0
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    ckpt = args.cfg["ckpt"]
    os.makedirs(ckpt, exist_ok=True)
    torch.save(model.state_dict(), f"{ckpt}/model.pt")
    print(f"saved {ckpt}/model.pt after {step} steps in {(time.time()-t0)/60:.1f} min")


@torch.no_grad()
def run_eval(args) -> None:
    dev = device()
    tok = tagger.get_tokenizer()
    model = tagger.SpanUnitTagger().to(dev)
    model.load_state_dict(torch.load(f"{args.cfg['ckpt']}/model.pt", map_location=dev))
    model.eval()

    recs = list(parse.load("data/train.csv"))
    assign = splits.assign(recs)
    val = [r for r in recs if assign[r.sample_id] == args.split and ev.gold_of(r)]
    cache = O.load_cache("cache/ocr_train.jsonl") if args.cfg["ocr"] else {}
    name = "S3 (+OCR)" if args.cfg["ocr"] else "S1"
    print(f"evaluating {name} and S0 on {len(val):,} {args.split} rows")

    rep1, rep0, repc = ev.Report(), ev.Report(), ev.Report()
    fellback = 0
    B = 64
    for i in range(0, len(val), B):
        chunk = val[i:i + B]
        texts = [label_spans.build_text(r, cache.get(r.image_file)) for r in chunk]
        enc = tok(texts, truncation=True, max_length=tagger.MAX_LEN, padding="max_length",
                  return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(dev) for k, v in enc.items()}
        tag_logits, unit_logits = model(enc["input_ids"], enc["attention_mask"])
        tags = tag_logits.argmax(-1).cpu().tolist()
        units = unit_logits.argmax(-1).cpu().tolist()
        for r, text, t, o, u in zip(chunk, texts, tags, offs.tolist(), units):
            gold = ev.gold_of(r)
            p1 = tagger.decode_prediction(text, t, o, u)
            p0 = rules.predict(r)
            rep1.add(r, p1, gold)
            rep0.add(r, p0, gold)
            # S1+S0 cascade: the tagger is far more accurate when it commits,
            # but it was trained only on solvable rows so it declines far more
            # often.  Fall back to the rules exactly where it declines.
            if p1.abstained:
                fellback += 1
                repc.add(r, p0, gold)
            else:
                repc.add(r, p1, gold)
        if i % (B * 20) == 0:
            print(f"  {i:,}/{len(val):,}", flush=True)

    print(rep0.table(f"S0 rule baseline - {args.split}"))
    print(rep1.table(f"{name} multi-task tagger - {args.split}"))
    print(repc.table(f"{name}+S0 cascade - {args.split}"))
    print(f"\ncascade fell back to S0 on {fellback:,} rows "
          f"({100*fellback/rep1.n:.1f}%)\n")
    for label, rep in ((name, rep1), ("cascade", repc)):
        d = 100 * (rep.hits["value"] - rep0.hits["value"]) / rep0.n
        ds = 100 * (rep.hits["strict"] - rep0.hits["strict"]) / rep0.n
        print(f"{label:<10} - S0:  value {d:+.1f} pts   strict {ds:+.1f} pts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--variant", default="text", choices=list(VARIANTS))
    args = ap.parse_args()
    args.cfg = VARIANTS[args.variant]
    tagger.MAX_LEN = args.cfg["max_len"]
    if not args.eval_only:
        train(args)
    run_eval(args)


if __name__ == "__main__":
    main()
