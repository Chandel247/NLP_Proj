"""Run Vision OCR over every image in a split and cache the text.

Resumable and deduplicated: 75,000 train rows reference 72,287 distinct images,
and a multi-hour job must survive being interrupted, so completed images are
skipped on restart and results are appended as they finish.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _one(filename: str) -> tuple[str, str]:
    import ocr as O
    return filename, O.ocr_text(filename)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--out", default="cache/ocr_train.jsonl")
    ap.add_argument("--dataset", default="2025", choices=["2025", "2024", "ws2024"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import ocr as O
    if args.dataset == "ws2024":
        import amazon2024 as _a
        loader_fn = _a.load_working_set
    elif args.dataset == "2024":
        import amazon2024 as _a
        loader_fn = _a.load
    else:
        import parse as _p
        loader_fn = _p.load

    wanted, seen = [], set()
    for r in loader_fn(args.csv, limit=args.limit):
        f = r.image_file
        if f not in seen and O.image_path(f):
            seen.add(f)
            wanted.append(f)

    done = set(O.load_cache(args.out))
    todo = [f for f in wanted if f not in done]
    print(f"{len(wanted):,} distinct images; {len(done):,} already cached; "
          f"{len(todo):,} to do on {args.workers} workers", flush=True)
    if not todo:
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, n = time.time(), 0
    with open(args.out, "a") as fh, mp.Pool(args.workers) as pool:
        for filename, text in pool.imap_unordered(_one, todo, chunksize=32):
            fh.write(json.dumps({"image": filename, "text": text}) + "\n")
            n += 1
            if n % 2000 == 0:
                rate = n / (time.time() - t0)
                fh.flush()
                print(f"  {n:,}/{len(todo):,}  {rate:.0f} img/s  "
                      f"eta {(len(todo)-n)/rate/60:.0f} min", flush=True)
    dt = time.time() - t0
    print(f"done: {n:,} images in {dt/60:.1f} min ({n/dt:.0f} img/s) -> {args.out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
