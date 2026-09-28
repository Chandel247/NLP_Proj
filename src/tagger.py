"""S1 - a multi-task text tagger for net content.

Two heads on one encoder, because the task has two different shapes:

  token head     BIO over SIZE / UNIT / PACK.  Learns *which* number in the
                 title is the size - the exact thing S0 gets wrong 27.6% of the
                 time inside `literal_name`.
  unit head      the canonical gold unit for the whole listing.  Titles write
                 "Ounce" for fluid ounces on 10.5% of labelled rows, and no span
                 can disambiguate that; the product category can ("Taco Sauce"
                 is a volume).  This head is what lets S1 beat S0 on `strict`.

Decoding stays deterministic: read the spans, apply pack arithmetic, convert the
surface unit into the predicted canonical unit.  The network chooses; arithmetic
is never left to it.
"""
from __future__ import annotations

import json

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

import normalize as N
from label_spans import TAG2ID, TAGS
from rules import FAMILY_UNIT, ABSTAIN, Prediction

MODEL_NAME = "distilroberta-base"
MAX_LEN = 192

# Canonical units worth a class of their own; everything rarer folds into OTHER.
UNIT_CLASSES = ["ounce", "fluid_ounce", "count", "pound", "gram", "milliliter", "OTHER"]
UNIT2ID = {u: i for i, u in enumerate(UNIT_CLASSES)}

# The 2024 corpus reports in a different vocabulary - centimetres and inches
# dominate because half its rows are dimensions - so the head's label set is
# swappable rather than hard-coded.
UNIT_CLASSES_2024 = ["centimeter", "inch", "gram", "kilogram", "pound", "ounce",
                     "milligram", "volt", "watt", "milliliter", "fluid_ounce",
                     "foot", "millimeter", "liter", "ton", "OTHER"]


def set_unit_classes(classes: list[str]) -> None:
    global UNIT_CLASSES, UNIT2ID
    UNIT_CLASSES = list(classes)
    UNIT2ID = {u: i for i, u in enumerate(UNIT_CLASSES)}


def unit_class(u: str) -> int:
    return UNIT2ID.get(u, UNIT2ID["OTHER"])


class SpanUnitTagger(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        h = cfg.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.tag_head = nn.Linear(h, len(TAGS))
        self.unit_head = nn.Linear(h, len(UNIT_CLASSES))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out = self.dropout(out)
        # <s> is the sequence summary for RoBERTa-family encoders.
        return self.tag_head(out), self.unit_head(out[:, 0])


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def encode(rows: list[dict], tok, with_labels: bool = True) -> dict:
    """Tokenise and project character spans onto tokens via offset mapping."""
    enc = tok([r["text"] for r in rows], truncation=True, max_length=MAX_LEN,
              padding="max_length", return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    if not with_labels:
        return {**enc, "offsets": offsets}

    labels = torch.full(offsets.shape[:2], -100, dtype=torch.long)
    for i, r in enumerate(rows):
        spans = [(r["spans"]["size"], "SIZE"), (r["spans"]["unit"], "UNIT")]
        if r["spans"]["pack"]:
            spans.append((r["spans"]["pack"], "PACK"))
        for j, (s, e) in enumerate(offsets[i].tolist()):
            if enc["attention_mask"][i][j] == 0 or s == e:
                continue                      # padding or a special token
            labels[i][j] = TAG2ID["O"]
            for (cs, ce), name in spans:
                if s < ce and e > cs:         # any overlap
                    labels[i][j] = TAG2ID[("B-" if s <= cs else "I-") + name]
                    break
    units = torch.tensor([unit_class(r["gold_unit"]) for r in rows])
    return {**enc, "labels": labels, "unit_labels": units}


def decode_prediction(text: str, tag_ids: list[int], offsets, unit_id: int) -> Prediction:
    """Spans + predicted unit -> a (value, unit) prediction."""
    def collect(name: str) -> str:
        chunks, cur = [], []
        for tid, (s, e) in zip(tag_ids, offsets):
            if s == e:
                continue
            tag = TAGS[tid]
            if tag.endswith(name):
                if tag.startswith("B-") and cur:
                    chunks.append(cur); cur = []
                cur.append((s, e))
            elif cur:
                chunks.append(cur); cur = []
        if cur:
            chunks.append(cur)
        if not chunks:
            return ""
        first = chunks[0]
        return text[first[0][0]:first[-1][1]]

    size_txt, unit_txt, pack_txt = collect("SIZE"), collect("UNIT"), collect("PACK")
    try:
        size = float(size_txt.replace(",", "").strip())
    except ValueError:
        return ABSTAIN
    surface = N.normalize_unit(unit_txt)
    target = UNIT_CLASSES[unit_id]

    if surface is None:
        # No unit span was tagged, but the sequence head still names a unit and
        # is usually right.  Reading the number as already being in that unit
        # recovers rows that would otherwise abstain for want of two tokens.
        if target in N.UNITS:
            pack_txt2 = pack_txt
            try:
                p2 = float(pack_txt2.replace(",", "").strip()) if pack_txt2 else 1.0
            except ValueError:
                p2 = 1.0
            p2 = p2 if 1 <= p2 <= 500 else 1.0
            return Prediction(size * p2, target, "s1_headunit")
        return ABSTAIN

    if target == "OTHER":
        target = FAMILY_UNIT.get(N.family(surface) or "", "")
    if target not in N.UNITS:
        return ABSTAIN

    value, bridged = N.catalog_convert(size, surface, target)
    if value is None:
        return ABSTAIN

    pack = 1.0
    if pack_txt:
        try:
            p = float(pack_txt.replace(",", "").strip())
            if 1 <= p <= 500:
                pack = p
        except ValueError:
            pass
    rule = "s1" + ("_x_pack" if pack != 1 else "") + ("_bridged" if bridged else "")
    return Prediction(value * pack, target, rule)


def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
