"""The full 2024 system on one (image, attribute) query, stage by stage.

This is the evaluated system, not a re-implementation of it: OCR, the S0 rules,
the tagger + rules cascade (S2), the LoRA selector (S3) and the pre-specified
router (S5) are the same code, checkpoints, prompt and image preprocessing that
produced the reported numbers.  `run` yields one event per stage so a front end
can show the pipeline filling in; `parity.py` checks it reproduces the results.
"""
from __future__ import annotations

import io
import os
import string
import sys
import time

# Every model is already in the local HF cache.  Without this, loading still asks the
# Hub for file metadata and hangs whenever the Hub throttles or the room has no Wi-Fi.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

import amazon2024 as A
import evaluate as ev
import label_spans
import normalize as N
import ocr as O
import rules2024
import tagger
from label_spans import TAGS

tagger.MAX_LEN = 224                       # as in route_2024.py / score_test_2024.py
tagger.set_unit_classes(tagger.UNIT_CLASSES_2024)

TAGGER_CKPT = os.path.join(ROOT, "cache/t2024/model.pt")
ADAPTER = os.path.join(ROOT, "kaggle/qwen2vl-dim-lora")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LENGTH = ("depth", "width", "height")
ATTRIBUTES = tuple(rules2024.ATTR_FAMILY)

# The selector was trained on images re-encoded at 768 px / q85 (build_lora_package)
# and fed through the processor at 512 x 512 max.  Both steps are reproduced.
CAP, Q = 768, 85
MAX_PIXELS = 512 * 512

# Verbatim from the training notebook - the prompt must match character for character.
PHRASE = {"depth": "depth (front to back)",
          "width": "width (left to right)",
          "height": "height (top to bottom)"}
SELECT = ("This product photograph shows measurements. Text recognition read "
          "these values from the image:\n{options}\n"
          "Which one is the product's {phrase}?\n"
          "Answer with only the letter.")


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def length_candidates(ocr_text: str) -> list[tuple[float, str]]:
    """Deduped length candidates in OCR order, capped at 8 - the option list the
    selector saw (build_lora_package.candidates_for, minus the gold bookkeeping)."""
    seen, uniq = set(), []
    for v, u, _ in rules2024.candidates(O.clean(ocr_text), "length"):
        k = round(N.to_base(v, u) or 0, 4)
        if k and k not in seen:
            seen.add(k)
            uniq.append((v, u))
    return uniq[:8]


def render_prompt(attribute: str, cands) -> str:
    opts = "\n".join(f"  ({string.ascii_lowercase[j]}) {v:g} {u}" for j, (v, u) in enumerate(cands))
    return SELECT.format(options=opts, phrase=PHRASE[attribute])


def training_view(img: Image.Image) -> Image.Image:
    """The 768 px q85 JPEG round-trip every training and eval image went through."""
    im = img.convert("RGB")
    im.thumbnail((CAP, CAP), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=Q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _pred(p) -> dict | None:
    return None if p.abstained else {"value": p.value, "unit": p.unit, "rule": p.rule}


class Pipeline:
    def __init__(self, *, load_vlm: bool = True, dtype=torch.float16):
        self.dev = device()
        self.tok = tagger.get_tokenizer()
        self.tagger = tagger.SpanUnitTagger().to(self.dev)
        self.tagger.load_state_dict(torch.load(TAGGER_CKPT, map_location=self.dev))
        self.tagger.eval()
        self.vlm = self.proc = None
        if load_vlm:
            self._load_vlm(dtype)

    def _load_vlm(self, dtype) -> None:
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=MAX_PIXELS)
        base = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, ADAPTER)
        n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        assert n_lora == 18_464_768, f"adapter did not load as trained ({n_lora:,} LoRA params)"
        # Merged, the adapter costs nothing per call; the weights are identical.
        self.vlm = model.merge_and_unload().to(self.dev).eval()
        self.letter_ids = {}
        for L in "abcdefgh":
            ids = self.proc.tokenizer.encode(L, add_special_tokens=False)
            assert len(ids) == 1, f"letter {L} is not a single token: {ids}"
            self.letter_ids[L] = ids[0]

    # --- stages -----------------------------------------------------------
    def cascade(self, attribute: str, ocr_text: str) -> dict:
        """S2: the tagger, falling back to the S0 rules when it abstains."""
        rec = A.Record2024(sample_id="query", image_link="", entity_name=attribute,
                           entity_value="", group_id="")
        text = label_spans.build_text(rec, ocr_text)
        enc = self.tok([text], truncation=True, max_length=tagger.MAX_LEN, padding="max_length",
                       return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.dev) for k, v in enc.items()}
        with torch.no_grad():
            tg, un = self.tagger(enc["input_ids"], enc["attention_mask"])
        tags = tg.argmax(-1)[0].cpu().tolist()
        unit_id = int(un.argmax(-1)[0])
        pt = tagger.decode_prediction(text, tags, offs, unit_id)
        ps = rules2024.predict(attribute, O.clean(ocr_text))
        chosen = ps if pt.abstained else pt
        spans = []
        for tid, (s, e) in zip(tags, offs):
            if s == e or TAGS[tid] == "O":
                continue
            kind = TAGS[tid][2:]
            if spans and spans[-1]["kind"] == kind and TAGS[tid].startswith("I-"):
                spans[-1]["end"] = e
            else:
                spans.append({"kind": kind, "start": s, "end": e})
        return {"input": text, "spans": spans, "unit_head": tagger.UNIT_CLASSES[unit_id],
                "tagger": _pred(pt), "rules": _pred(ps), "cascade": _pred(chosen),
                "source": "rules" if pt.abstained else "tagger", "_pred": chosen}

    def select(self, img: Image.Image, attribute: str, cands) -> dict:
        """S3: the fine-tuned selector.  One forward pass; the first token is the
        decision, which is what greedy decoding took in every Kaggle run."""
        prompt = render_prompt(attribute, cands)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[chat], images=[img], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            logits = self.vlm(**inp).logits[0, -1].float()
        logp = torch.log_softmax(logits, -1)
        letters = "abcdefgh"[:len(cands)]
        lp = torch.stack([logp[self.letter_ids[L]] for L in letters])
        probs = torch.softmax(lp, -1).tolist()
        top = int(logits.argmax())
        raw = self.proc.tokenizer.decode([top]).strip()
        j = letters.find(raw.lower()) if len(raw) == 1 else -1
        return {"prompt": prompt, "letters": list(letters), "probs": [round(p, 5) for p in probs],
                "letter_mass": round(float(lp.exp().sum()), 5), "raw": raw,
                "pick": letters[j] if j >= 0 else None,
                "pred": list(cands[j]) if j >= 0 else None,
                "visual_tokens": int(inp["image_grid_thw"].prod()) // 4}

    # --- the whole system ---------------------------------------------------
    def run(self, attribute: str, *, image_path: str | None = None, image: Image.Image | None = None,
            ocr_text: str | None = None, ocr_lines=None, control: str = "real", gold=None):
        """Yield one event per stage.  Pass `ocr_text` to reuse cached OCR (parity)."""
        if attribute not in ATTRIBUTES:
            raise ValueError(f"unknown attribute {attribute!r}")
        t = time.perf_counter()

        def ms() -> int:
            nonlocal t
            now = time.perf_counter(); d = now - t; t = now
            return round(1000 * d)

        if ocr_text is None:
            ocr_lines = O.ocr_boxes(image_path)
            ocr_text = "\n".join(s for s, _ in ocr_lines)
            yield {"stage": "ocr", "ms": ms(), "source": "Apple Vision",
                   "lines": [{"text": s, "box": [round(c, 4) for c in b]} for s, b in ocr_lines]}
        else:
            yield {"stage": "ocr", "ms": ms(), "source": "cache",
                   "lines": [{"text": s, "box": [round(c, 4) for c in b]} for s, b in (ocr_lines or [])]
                   or [{"text": s, "box": None} for s in ocr_text.split("\n")]}

        family = rules2024.ATTR_FAMILY[attribute]
        cleaned = O.clean(ocr_text)
        cands = rules2024.candidates(cleaned, family)
        yield {"stage": "candidates", "ms": ms(), "family": family, "cleaned": cleaned,
               "items": [{"value": v, "unit": u, "pos": p} for v, u, p in cands]}

        cas = self.cascade(attribute, ocr_text)
        chosen = cas.pop("_pred")
        yield {"stage": "cascade", "ms": ms(), **cas}

        final, source, reason = chosen, "cascade", "not a length attribute: the cascade answers"
        if attribute in LENGTH:
            opts = length_candidates(ocr_text)
            if not opts:
                reason = "OCR found no length candidate for the VLM to choose from: cascade fallback"
                yield {"stage": "vlm", "ms": ms(), "skipped": "no candidates", "options": []}
            elif len(opts) == 1:
                final = _single(opts[0])
                source, reason = "vlm", "one candidate: nothing to choose, it is the answer"
                yield {"stage": "vlm", "ms": ms(), "skipped": "single candidate",
                       "options": [list(o) for o in opts]}
            elif self.vlm is None:
                reason = "VLM not loaded: cascade fallback"
                yield {"stage": "vlm", "ms": ms(), "skipped": "VLM not loaded",
                       "options": [list(o) for o in opts]}
            else:
                img = training_view(image if image is not None else Image.open(image_path))
                if control == "blank":
                    img = Image.new("RGB", img.size, "white")
                sel = self.select(img, attribute, opts)
                yield {"stage": "vlm", "ms": ms(), "control": control,
                       "options": [list(o) for o in opts], **sel}
                if sel["pred"] is not None:
                    final = _single(tuple(sel["pred"]))
                    source, reason = "vlm", f"VLM chose ({sel['pick']}) among {len(opts)} candidates"
                else:
                    reason = "VLM gave no valid letter: cascade fallback"

        out = {"stage": "final", "ms": ms(), "source": source, "reason": reason,
               "answer": None if final.abstained else {"value": final.value, "unit": final.unit}}
        if gold is not None:
            lo, hi, gu = gold
            out["gold"] = {"low": lo, "high": hi, "unit": gu}
            out["correct"] = bool(ev.score(final, gold)["dim"])
        yield out

    def answer(self, attribute: str, **kw) -> dict:
        """Run to completion and return the final event."""
        for ev_ in self.run(attribute, **kw):
            pass
        return ev_


def _single(vu):
    from rules import Prediction
    return Prediction(float(vu[0]), vu[1], "s3_select")
