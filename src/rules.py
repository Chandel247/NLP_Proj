"""S0 - the rule-based baseline, and the silver labeller for Track B.

This is deliberately strong.  44.5% of targets appear verbatim in the item name
and another 25.4% are a product of two numbers there, so a careful rule system
sets a high bar that every neural model must clear to justify itself.

At inference the model sees neither Value nor Unit, so S0 must predict the unit
too.  It infers the dimensional *family* from the text, then emits that family's
conventional catalogue unit (mass -> ounce, volume -> fluid_ounce, count ->
count), which is how the corpus overwhelmingly reports each family.
"""
from __future__ import annotations

from dataclasses import dataclass

import normalize as N

# The unit each family is conventionally reported in by this catalogue.
FAMILY_UNIT = {"mass": "ounce", "volume": "fluid_ounce", "count": "count"}


@dataclass
class Prediction:
    value: float | None
    unit: str | None
    rule: str            # which branch fired - drives the error analysis

    @property
    def abstained(self) -> bool:
        return self.value is None or self.unit is None


ABSTAIN = Prediction(None, None, "abstain")


def _emit(qty: N.Quantity, pack: int | None, rule: str) -> Prediction:
    fam = N.family(qty.unit)
    target = FAMILY_UNIT.get(fam)
    if target is None:
        return ABSTAIN
    val = N.convert(qty.value, qty.unit, target)
    if val is None:
        return ABSTAIN
    return Prediction(val * (pack or 1), target, rule)


def predict(rec) -> Prediction:
    """Predict (total value, canonical unit) from text alone."""
    name = rec.item_name
    pack = N.find_pack_count(name)
    qs = N.find_quantities(name, source="item_name")

    # Prefer a mass/volume size over a bare count: "12 Ounce (Pack of 6)" is a
    # size times a pack, whereas the 6 alone is the pack, not the answer.
    sized = [q for q in qs if N.family(q.unit) in ("mass", "volume")]
    if sized:
        return _emit(sized[0], pack, "name_size" + ("_x_pack" if pack else ""))

    counts = [q for q in qs if q.unit == "count"]
    if counts:
        return Prediction(counts[0].value, "count", "name_count")

    # Nothing sized in the title - fall back to bullets, then description.
    for source, text in (("bullets", "\n".join(rec.bullets)), ("description", rec.description)):
        if not text:
            continue
        qs2 = [q for q in N.find_quantities(text, source=source)
               if N.family(q.unit) in ("mass", "volume")]
        if qs2:
            return _emit(qs2[0], pack, f"{source}_size" + ("_x_pack" if pack else ""))

    # A pack count with no size at all is a count-type product ("Pack of 6").
    if pack:
        return Prediction(float(pack), "count", "pack_only")

    return ABSTAIN
