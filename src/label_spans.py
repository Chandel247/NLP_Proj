"""Weak supervision: turn a (value, unit) label into token spans.

We have no span annotations, only the total.  But the total is a *program* over
the text - pick a size, pick its unit, optionally multiply by a pack count - so
we can search for the assignment that reproduces the gold and take that as the
label.  This is what teaches S1 the thing S0 gets wrong: 45.7% of answers sit
verbatim in the title, yet the rules score only 72.4% there because they choose
the wrong number.  Here the choice is supervised.

A row yields labels only when the search finds a consistent assignment, so the
`text_unrecoverable` bucket contributes nothing - correctly, since no span in
the text explains its answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import normalize as N

TOL = 0.01
MAX_CHARS = 600
OCR_CHARS = 400
OCR_MARK = "\nPKG "     # marks the start of the on-package channel

# Integers that could serve as a pack multiplier, with their character spans.
_INT_RE = re.compile(r"(?<![\w.])(\d{1,3})(?![\d.])")
# Explicit pack phrasing - a multiplier found here is far more trustworthy than
# an arbitrary integer that happens to make the arithmetic work.
_PACK_CTX_RE = re.compile(
    r"(?:\(\s*)?(?:pack|set|case|box|bundle|lot)\s+of\s+(\d{1,3})\b"
    r"|\b(\d{1,3})\s*[-\s]?\s*(?:pack|pk|pc\b|pcs\b|count\b|ct\b)",
    re.I,
)

Span = tuple[int, int]


@dataclass
class Assignment:
    size: Span
    unit: Span
    pack: Span | None
    tier: int          # 0 = simplest explanation; higher = more machinery
    bridged: bool      # the ounce/fluid-ounce crossing was needed

    def as_dict(self) -> dict:
        return {"size": list(self.size), "unit": list(self.unit),
                "pack": list(self.pack) if self.pack else None}


def _close(a: float, b: float) -> bool:
    return b != 0 and abs(a - b) / abs(b) <= TOL


def _pack_candidates(text: str) -> list[tuple[float, Span, bool]]:
    """(value, span, from_explicit_pack_phrase) for every plausible multiplier."""
    explicit: dict[Span, float] = {}
    for m in _PACK_CTX_RE.finditer(text):
        g = 1 if m.group(1) else 2
        explicit[(m.start(g), m.end(g))] = float(m.group(g))
    out = [(v, s, True) for s, v in explicit.items()]
    for m in _INT_RE.finditer(text):
        span = (m.start(1), m.end(1))
        if span in explicit:
            continue
        v = float(m.group(1))
        if 2 <= v <= 500:                 # 1 is the implicit no-pack case
            out.append((v, span, False))
    return out


def find_assignment(text: str, gold_value: float, gold_unit: str) -> Assignment | None:
    """Search for size x pack -> gold, preferring the simplest explanation."""
    quantities = N.find_quantities(text)

    sizes: list[tuple[float, str, Span, Span]] = []
    for q in quantities:
        # `find_quantities` spans cover "12 Ounce" as a whole; split it so the
        # number and the unit carry different tags.  Both sides are matched
        # explicitly - a lazy quantifier here silently clips "12" to "1", and a
        # span that keeps its leading space shifts the unit's first token off
        # its B- tag.
        m = re.match(r"([\d][\d.,]*)\s*[-‐-―]?\s*(\S.*?)\s*$", text[q.start:q.end])
        if not m:
            continue
        num_span = (q.start + m.start(1), q.start + m.end(1))
        unit_span = (q.start + m.start(2), q.start + m.end(2))
        sizes.append((q.value, q.unit, num_span, unit_span))

    packs = _pack_candidates(text)
    best: Assignment | None = None

    for value, unit, num_span, unit_span in sizes:
        conv, bridged = N.catalog_convert(value, unit, gold_unit)
        if conv is None:
            continue
        # tier 0: the size alone is the answer.
        if _close(conv, gold_value):
            cand = Assignment(num_span, unit_span, None, 0, bridged)
            if best is None or cand.tier < best.tier:
                best = cand
            continue
        for pv, pspan, explicit in packs:
            if pspan == num_span:                 # a number cannot be both
                continue
            if _close(conv * pv, gold_value):
                tier = 1 if explicit else 2
                cand = Assignment(num_span, unit_span, pspan, tier, bridged)
                if best is None or tier < best.tier:
                    best = cand
    return best


def build_text(rec, ocr_raw: str | None = None) -> str:
    """The exact string a model sees.  Single source of truth for train and eval."""
    import ocr as O
    text = rec.model_input(include_description=False)[:MAX_CHARS]
    if ocr_raw:
        salient = O.salient(O.clean(ocr_raw), OCR_CHARS)
        if salient:
            text += OCR_MARK + salient
    return text


def build(rec, gold, ocr_raw: str | None = None) -> dict | None:
    """One training example, or None when no assignment explains the label.

    `gold` is (low, high, unit); the search targets the interval midpoint for a
    range label, which is exact for the scalar case where low == high.
    """
    text = build_text(rec, ocr_raw)
    if not text:
        return None
    lo, hi, gu = gold
    gv = (lo + hi) / 2 if hi != lo else lo
    a = find_assignment(text, gv, gu)
    if a is None:
        return None
    return {"sample_id": rec.sample_id, "text": text, "spans": a.as_dict(),
            "tier": a.tier, "bridged": a.bridged, "gold_value": gv, "gold_unit": gu,
            "has_ocr": bool(ocr_raw)}


TAGS = ["O", "B-SIZE", "I-SIZE", "B-UNIT", "I-UNIT", "B-PACK", "I-PACK"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}


def write_jsonl(path: str, rows) -> int:
    n = 0
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n
