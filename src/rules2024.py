"""S0 for the 2024 task: pick the right quantity off the packaging.

The task is conditional - the attribute name is an input - so the first move is
dimensional: `wattage` can only be answered by a power quantity, `item_volume`
only by a volume one. That alone discards most OCR noise, since a barcode or a
batch code carries no unit.

What remains is choosing among same-family candidates ("NET WT 12 OZ (340g)"
offers two), which is exactly the decision a learned tagger should make better
than a rule - the same finding the 2025 corpus produced.
"""
from __future__ import annotations

import re

import normalize as N
from rules import ABSTAIN, Prediction

ATTR_FAMILY = {
    "item_weight": "mass",
    "maximum_weight_recommendation": "mass",
    "item_volume": "volume",
    "voltage": "voltage",
    "wattage": "power",
    # depth / width / height are 50.3% of the corpus and share one family, so
    # the dimensional filter that carries the other five attributes cannot
    # separate them at all - "12 x 8 x 3 cm" offers three length values and
    # only a word cue says which is which.  Expect S0 to be weak here.
    "depth": "length",
    "width": "length",
    "height": "length",
}

# Per-attribute word cues, needed only where the family is ambiguous.
_DIM_CUE = {
    "depth": re.compile(r"\bdepth\b|\bdeep\b|\bD\s*[:=]|\bthick", re.I),
    "width": re.compile(r"\bwidth\b|\bwide\b|\bW\s*[:=]|\bbreadth", re.I),
    "height": re.compile(r"\bheight\b|\bhigh\b|\bH\s*[:=]|\btall", re.I),
}

# Packaging marks the declared net quantity explicitly; a candidate sitting
# near one of these is far more likely to be the answer than a stray figure.
_NET_CUE = re.compile(r"NET\s*WT|NET|CONTENTS|CAPACITY|VOL\b", re.I)
_MAX_CUE = re.compile(r"MAX|CAPACITY|UP\s*TO|LIMIT|LOAD", re.I)

HEURISTICS = ("first", "largest", "smallest", "cued")


def candidates(text: str, family: str) -> list[tuple[float, str, int]]:
    """(value, canonical unit, position) for every same-family quantity."""
    out = [(q.value, q.unit, q.start) for q in N.find_quantities(text)
           if N.family(q.unit) == family]
    # A shared-unit range contributes its low end: "100-240V" on a universal
    # supply is reported by the catalogue as the range's start more often than
    # its end.
    out += [(lo, u, st) for lo, hi, u, st, _ in N.find_ranges(text)
            if N.family(u) == family]
    return out


def predict(entity_name: str, text: str, heuristic: str = "cued") -> Prediction:
    family = ATTR_FAMILY.get(entity_name)
    if family is None or not text:
        return ABSTAIN
    cands = candidates(text, family)
    if not cands:
        return ABSTAIN

    if heuristic == "first":
        value, unit, _ = cands[0]
    elif heuristic == "largest":
        value, unit, _ = max(cands, key=lambda c: N.to_base(c[0], c[1]) or 0)
    elif heuristic == "smallest":
        value, unit, _ = min(cands, key=lambda c: N.to_base(c[0], c[1]) or 0)
    else:                                    # "cued"
        cue = (_DIM_CUE.get(entity_name)
               or (_MAX_CUE if entity_name == "maximum_weight_recommendation" else _NET_CUE))
        marks = [m.start() for m in cue.finditer(text)]
        if marks:
            value, unit, _ = min(
                cands, key=lambda c: min(abs(c[2] - m) for m in marks))
        else:
            value, unit, _ = cands[0]
    return Prediction(value, unit, f"s0_2024_{heuristic}")
