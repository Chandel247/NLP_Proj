"""Leakage-safe train/val/test splits.

Amazon catalogues are dense with near-duplicates: the same product listed at six
pack sizes, or two sellers sharing one product photo.  A random row split puts
those siblings on both sides and inflates every metric.  We therefore build
groups with a union-find over two signals - a size-stripped item-name signature
and the image file - and assign whole groups to splits.
"""
from __future__ import annotations

import hashlib
import re

from normalize import TEXT_ALIASES

_UNIT_TOK = "|".join(sorted((re.escape(a) for a in TEXT_ALIASES), key=len, reverse=True))
_STRIP_RE = re.compile(rf"\b(?:{_UNIT_TOK})\b|\d+(?:\.\d+)?|\bpack of\b|\bset of\b", re.I)
_PUNCT_RE = re.compile(r"[^a-z ]+")


def name_signature(item_name: str, n_tokens: int = 8) -> str:
    """Item name with sizes, counts and punctuation removed.

    "Salerno Butter Cookies, 8 Ounce (Pack of 4)" and the 12-ounce listing of
    the same product collapse onto one signature.
    """
    s = _STRIP_RE.sub(" ", item_name.lower())
    s = _PUNCT_RE.sub(" ", s)
    toks = s.split()[:n_tokens]
    return " ".join(toks)


class _DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_ids(records) -> dict[str, str]:
    """sample_id -> group id.  Rows sharing a name signature or an image merge."""
    dsu = _DSU()
    for r in records:
        node = f"s:{r.sample_id}"
        sig = name_signature(r.item_name)
        if sig:
            dsu.union(node, f"n:{sig}")
        dsu.union(node, f"i:{r.image_file}")
    return {r.sample_id: dsu.find(f"s:{r.sample_id}") for r in records}


def assign(records, ratios=(0.8, 0.1, 0.1), seed: int = 13) -> dict[str, str]:
    """sample_id -> 'train' | 'val' | 'test', hashed by group so it is stable.

    Hashing the group id (rather than shuffling) keeps assignments identical as
    rows are added or the corpus is re-subsampled.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9
    gid = group_ids(records)
    tr, va = ratios[0], ratios[0] + ratios[1]
    out = {}
    for sid, g in gid.items():
        h = hashlib.blake2b(f"{seed}:{g}".encode(), digest_size=8).digest()
        u = int.from_bytes(h, "big") / 2 ** 64
        out[sid] = "train" if u < tr else "val" if u < va else "test"
    return out
