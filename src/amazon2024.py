"""Amazon ML Challenge 2024 adapter.

The 2024 task is conditional and image-only: given a product photo *and* the
name of a wanted attribute, return its value.  There is no catalogue text at
all, so the text channel that carried ~83% of answers in the 2025 corpus does
not exist here - OCR is the only evidence, which is exactly why the S3
machinery ports and the S1 results do not.

Differences from the 2025 schema that the rest of the pipeline must absorb:
  * labels are `"750.0 watt"` or `"[110.0, 130.0] volt"` - 4.7% are genuine
    ranges, so gold is an interval throughout;
  * the requested attribute is part of the *input*, not a fixed target, so it
    is prepended to the model's text;
  * `group_id` identifies a product family and is the natural split key.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass

import normalize as N

ATTRIBUTES = ("item_weight", "item_volume", "maximum_weight_recommendation",
              "voltage", "wattage")

# "750.0 watt" | "[110.0, 130.0] volt" | "1.6 fluid ounce"
_SCALAR = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s+(.+?)\s*$")
_RANGE = re.compile(r"^\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s+(.+?)\s*$")


def parse_entity_value(raw: str) -> tuple[float, float, str] | None:
    """-> (low, high, canonical unit); low == high for a scalar label."""
    if not raw:
        return None
    m = _RANGE.match(raw)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3)
    else:
        m = _SCALAR.match(raw)
        if not m:
            return None
        lo = hi = float(m.group(1))
        unit = m.group(2)
    canon = N.normalize_unit(unit)
    if canon is None or hi < lo or hi <= 0:
        return None
    return lo, hi, canon


@dataclass
class Record2024:
    sample_id: str
    image_link: str
    entity_name: str
    entity_value: str
    group_id: str
    image_file_name: str = ""

    @property
    def image_file(self) -> str:
        """The rebuilt working set names images by their Amazon CDN basename
        (stable and unique); the older sheets used `<row_uid>.jpg`."""
        return self.image_file_name or f"{self.sample_id}.jpg"

    @property
    def gold(self) -> tuple[float, float, str] | None:
        return parse_entity_value(self.entity_value)

    # --- shims so the shared pipeline can consume these records unchanged ---
    @property
    def bullets(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return ""

    @property
    def item_name(self) -> str:
        return ""

    def model_input(self, *, include_description: bool = False) -> str:
        """The query half of the input: which attribute is being asked for.

        The OCR channel is appended by `label_spans.build_text`, so a full
        input reads `item_weight\\nPKG NET WT 12 OZ ...`.
        """
        return self.entity_name


def load(csv_path: str, limit: int | None = None):
    with open(csv_path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                break
            yield Record2024(
                sample_id=row["row_uid"],
                image_link=row["image_link"],
                entity_name=row["entity_name"],
                entity_value=row.get("entity_value", ""),
                group_id=row.get("group_id", ""),
            )


def load_working_set(csv_path: str, limit: int | None = None, split: str | None = None):
    """Load the rebuilt working set (natural attribute proportions, 8 attributes).

    Splits are precomputed in the file by `group_id`, so they are stable across
    runs and cannot drift between train and eval.
    """
    with open(csv_path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit is not None and i >= limit:
                break
            if split and row.get("split") != split:
                continue
            yield Record2024(
                sample_id=row["sample_id"],
                image_link=row["image_link"],
                entity_name=row["entity_name"],
                entity_value=row["entity_value"],
                group_id=row.get("group_id", ""),
                image_file_name=row.get("image_file", ""),
            )


def assign_splits(records, ratios=(0.8, 0.1, 0.1), seed: int = 13) -> dict[str, str]:
    """Split by `group_id` - a product family, so variants cannot straddle."""
    import hashlib
    tr, va = ratios[0], ratios[0] + ratios[1]
    out = {}
    for r in records:
        key = r.group_id or r.sample_id
        h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
        u = int.from_bytes(h, "big") / 2 ** 64
        out[r.sample_id] = "train" if u < tr else "val" if u < va else "test"
    return out
