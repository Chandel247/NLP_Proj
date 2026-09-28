"""Rebuild the 2024 working set from the full 263,859-row dataset.

The previously used sample was stratified ~equally across five attributes,
which distorted everything measured on it: `maximum_weight_recommendation` is
1.24% of the real data but 20% of that sample, and since it is the one attribute
with a 9% label-error rate, it inflated the apparent noise floor tenfold.  It
also dropped depth/width/height entirely - 50.4% of the corpus.

So: a simple random sample, which preserves natural proportions by construction,
over all eight attributes, split by `group_id` (750 of them, against 166 in the
old sample).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyarrow.parquet as pq

import amazon2024 as A

OUT_DIR = "data2024"


def image_basename(link: str) -> str:
    """Amazon CDN basenames are unique, so they double as a stable filename."""
    return link.rsplit("/", 1)[-1]


def split_of(group_id: str, ratios=(0.8, 0.1, 0.1), seed: int = 13) -> str:
    h = hashlib.blake2b(f"{seed}:{group_id}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2 ** 64
    return "train" if u < ratios[0] else "val" if u < ratios[0] + ratios[1] else "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000, help="rows to sample")
    ap.add_argument("--parquet", default=f"{OUT_DIR}/train.parquet")
    ap.add_argument("--out", default=f"{OUT_DIR}/working_set.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drop-attrs", default="",
                    help="comma-separated attributes to exclude entirely")
    args = ap.parse_args()

    t = pq.read_table(args.parquet).to_pydict()
    total = len(t["entity_name"])
    drop = {a for a in args.drop_attrs.split(",") if a}

    # Dedupe on (image, attribute): the same photo legitimately carries several
    # attributes, but the same pair twice is a duplicate.
    seen, pool = set(), []
    unparseable = dupes = dropped = 0
    for i in range(total):
        link, gid = t["image_link"][i], str(t["group_id"][i])
        attr, val = t["entity_name"][i], str(t["entity_value"][i])
        if attr in drop:
            dropped += 1
            continue
        if A.parse_entity_value(val) is None:
            unparseable += 1
            continue
        key = (link, attr)
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        pool.append((link, gid, attr, val))

    print(f"source rows            {total:,}")
    if drop:
        print(f"  excluded attributes  {dropped:,}  ({','.join(sorted(drop))})")
    print(f"  unparseable label    {unparseable:,}")
    print(f"  duplicate img+attr   {dupes:,}")
    print(f"  eligible             {len(pool):,}")

    random.seed(args.seed)
    sample = pool if args.n >= len(pool) else random.sample(pool, args.n)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for link, gid, attr, val in sample:
        sid = hashlib.blake2b(f"{link}|{attr}".encode(), digest_size=6).hexdigest()
        rows.append({"sample_id": sid, "image_link": link,
                     "image_file": image_basename(link), "group_id": gid,
                     "entity_name": attr, "entity_value": val,
                     "split": split_of(gid, seed=args.seed)})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"\nsampled {n:,} rows -> {args.out}")
    print(f"distinct images        {len({r['image_file'] for r in rows}):,}")
    print(f"distinct group_id      {len({r['group_id'] for r in rows}):,}")

    print(f"\n{'attribute':<32}{'sample':>9}{'share':>8}{'source share':>14}")
    print("-" * 63)
    src = Counter(t["entity_name"][i] for i in range(total))
    got = Counter(r["entity_name"] for r in rows)
    for a in sorted(got, key=lambda x: -got[x]):
        print(f"{a:<32}{got[a]:>9,}{100*got[a]/n:>7.2f}%{100*src[a]/total:>13.2f}%")

    print(f"\n{'split':<10}{'rows':>9}{'share':>8}{'groups':>9}")
    print("-" * 36)
    for s in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == s]
        print(f"{s:<10}{len(sel):>9,}{100*len(sel)/n:>7.1f}%"
              f"{len({r['group_id'] for r in sel}):>9}")


if __name__ == "__main__":
    main()
