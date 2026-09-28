"""Fetch working-set images from the Amazon CDN.

Resumable and idempotent: already-present files are skipped, so an interrupted
run costs nothing.  Concurrency is kept modest deliberately - this is a public
CDN being read for a course project, not a target.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
OUT = "data2024/images"


def fetch(job: tuple[str, str]) -> tuple[str, str | None]:
    filename, url = job
    dst = os.path.join(OUT, filename)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return filename, None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if not data:
                return filename, "empty"
            tmp = dst + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, dst)
            return filename, None
        # Catch broadly and on purpose.  A truncated CDN response raises
        # http.client.IncompleteRead, which is an HTTPException and not an
        # OSError - narrow handling let a single bad response kill the whole
        # pool 12,855 images into a 24,336-image run.  In a bulk fetcher, any
        # per-item failure must be recorded and stepped over, never raised.
        except Exception as e:
            if attempt == 2:
                return filename, type(e).__name__
            time.sleep(1.0)
    return filename, "unreachable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data2024/working_set.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    jobs: dict[str, str] = {}
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            jobs.setdefault(r["image_file"], r["image_link"])
    todo = [(f, u) for f, u in jobs.items()
            if not (os.path.exists(os.path.join(OUT, f))
                    and os.path.getsize(os.path.join(OUT, f)) > 0)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(jobs):,} distinct images; {len(jobs)-len(todo):,} present; "
          f"{len(todo):,} to fetch on {args.workers} workers", flush=True)

    fails: dict[str, str] = {}
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for filename, err in pool.map(fetch, todo):
            done += 1
            if err:
                fails[filename] = err
            if done % 2000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done:,}/{len(todo):,}  {rate:.0f} img/s  "
                      f"eta {(len(todo)-done)/rate/60:.0f} min  fails {len(fails):,}", flush=True)

    dt = time.time() - t0
    print(f"\ndone: {done-len(fails):,} fetched, {len(fails):,} failed "
          f"in {dt/60:.1f} min ({done/max(dt,1):.0f} img/s)")
    if fails:
        from collections import Counter
        for k, c in Counter(fails.values()).most_common():
            print(f"  {k}: {c:,}")
        with open("data2024/image_failures.txt", "w") as fh:
            for f, e in fails.items():
                fh.write(f"{f}\t{e}\n")
        print("  wrote data2024/image_failures.txt")


if __name__ == "__main__":
    main()
