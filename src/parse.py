"""Parse the semi-structured `catalog_content` blob into typed fields.

`catalog_content` is a flat block of `Key: value` lines where values may wrap
across many lines.  Only five base keys ever occur (optionally numbered):

    Item Name / Bullet Point / Product Description / Value / Unit

`Value` and `Unit` are the extraction TARGETS, so `model_input()` strips them;
anything that reads them for supervision must go through `gold()`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# A new field starts at a line beginning with one of the known keys, optionally
# numbered ("Bullet Point 12:").  Matching the closed set rather than a generic
# `\w+:` keeps colons inside prose (e.g. "Note: keep frozen") from splitting a value.
_KEYS = r"Item Name|Bullet Point|Product Description|Value|Unit"
_FIELD_RE = re.compile(rf"^({_KEYS})\s*\d*\s*:[ \t]?", re.M)

TARGET_KEYS = {"Value", "Unit"}


@dataclass
class Record:
    sample_id: str
    image_link: str
    item_name: str = ""
    bullets: list[str] = field(default_factory=list)
    description: str = ""
    raw_value: str | None = None
    raw_unit: str | None = None
    price: float | None = None

    @property
    def image_file(self) -> str:
        """Basename used on disk; image dirs are flat."""
        return self.image_link.rsplit("/", 1)[-1]

    def model_input(self, *, include_description: bool = True) -> str:
        """The text a model is allowed to see - targets removed."""
        parts = [self.item_name] + self.bullets
        if include_description:
            parts.append(self.description)
        return "\n".join(p for p in parts if p).strip()


def split_fields(blob: str) -> list[tuple[str, str]]:
    """Split into (base_key, value) pairs, preserving document order."""
    out: list[tuple[str, str]] = []
    matches = list(_FIELD_RE.finditer(blob))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
        out.append((m.group(1), blob[m.end():end].strip()))
    return out


def parse_catalog(blob: str) -> dict:
    """Group the field pairs by base key."""
    rec = {"item_name": "", "bullets": [], "description": "", "raw_value": None, "raw_unit": None}
    for key, val in split_fields(blob):
        if not val:
            continue
        if key == "Item Name":
            # Numbered variants exist (15 rows); keep the first, richest one.
            rec["item_name"] = rec["item_name"] or val
        elif key == "Bullet Point":
            rec["bullets"].append(val)
        elif key == "Product Description":
            rec["description"] = "\n".join(filter(None, [rec["description"], val]))
        elif key == "Value":
            rec["raw_value"] = val
        elif key == "Unit":
            rec["raw_unit"] = val
    return rec


def to_float(s: str | None) -> float | None:
    """`Value` is occasionally the literal string 'nan'."""
    if s is None:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def load(path: str, limit: int | None = None):
    """Stream Records from a train/test CSV.  `price` is absent in test.csv."""
    import csv
    csv.field_size_limit(10 ** 9)
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                break
            yield Record(
                sample_id=row["sample_id"],
                image_link=row["image_link"],
                price=to_float(row.get("price")),
                **parse_catalog(row["catalog_content"]),
            )
