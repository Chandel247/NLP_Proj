"""On-package text via Apple's Vision framework.

The attribute we want is printed on the packaging, so the useful visual signal
here is literally text.  Vision ships with macOS: no model download, no GPU, no
API.  Language correction is disabled deliberately - it is tuned for prose and
happily "corrects" the very tokens we need ("16 OZ" -> "16 O2", "1.9" -> "19").
"""
from __future__ import annotations

import json
import os

# Train and test images live in differently-named flat directories.
IMAGE_DIRS = (
    "data/train/images", "data/test/test", "data/images",          # 2025
    "data2024/images",                                                                     # 2024 rebuilt
    os.path.expanduser("~/Multimodal_Product_Attribution_Extraction_SHARE/data/images"),  # 2024 old sample
)


def image_path(filename: str) -> str | None:
    for d in IMAGE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def ocr(path: str, *, fast: bool = False) -> list[str]:
    """Recognised text lines, top candidate each, in reading order.

    Vision is imported lazily so that `clean`, `salient`, `load_cache` and
    `image_path` stay importable off macOS - training runs on Colab read the
    cached OCR and must never need pyobjc.
    """
    return [text for text, _ in ocr_boxes(path, fast=fast)]


def ocr_boxes(path: str, *, fast: bool = False) -> list[tuple[str, tuple[float, float, float, float]]]:
    """As `ocr`, with each line's box as (x, y, w, h) in 0-1 image fractions,
    origin top-left.  Vision reports boxes with a bottom-left origin."""
    import Quartz
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(path)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        return []
    image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if image is None:
        return []

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelFast if fast
        else Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(["en-US"])

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    ok, _ = handler.performRequests_error_([request], None)
    if not ok:
        return []
    results = request.results() or []
    lines = []
    for obs in results:
        cand = obs.topCandidates_(1)
        if cand:
            b = obs.boundingBox()
            x, y, w, h = b.origin.x, b.origin.y, b.size.width, b.size.height
            lines.append((str(cand[0].string()), (x, 1.0 - y - h, w, h)))
    return lines


def ocr_text(filename: str, *, fast: bool = False) -> str:
    p = image_path(filename)
    return "\n".join(ocr(p, fast=fast)) if p else ""


def load_cache(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["image"]] = r["text"]
    return out


# Vision confuses letters and digits on packaging type: "12 OZ" comes back as
# "12 0Z", "16 FL OZ" as "16 FLOZ".  Left alone these silently fail unit
# matching, which would look like "the image has no size on it".  Only
# digit-adjacent forms are rewritten, so a real number is never damaged.
import re as _re

_FIXES = (
    (_re.compile(r"(\d\s*)0Z\b", _re.I), r"\1OZ"),
    (_re.compile(r"(\d\s*)O2\b", _re.I), r"\1OZ"),
    (_re.compile(r"(\d\s*)0z\b"), r"\1oz"),
    (_re.compile(r"\bFLOZ\b", _re.I), "FL OZ"),
    (_re.compile(r"\bFL\.?\s*0Z\b", _re.I), "FL OZ"),
    (_re.compile(r"(\d)\s*[lI]b\b"), r"\1 lb"),
    (_re.compile(r"NET\s*(?:WT|WEIGHT|HT)\.?", _re.I), "NET WT"),
    # "DC12V" / "AC220V": the quantity regex refuses a number preceded by a
    # letter, so these power-supply prefixes hide the value entirely.
    (_re.compile(r"\b(AC|DC)(?=\d)", _re.I), r"\1 "),
    # European decimal comma: "7,9 g" and "0,28 oz" are 7.9 and 0.28, but the
    # number regex reads them as 9 and 28 - a 100x error that looks like a bad
    # label.  One or two digits after the comma is a decimal; three is a
    # thousands separator ("1,234 gram") and is left alone.
    (_re.compile(r"(\d),(\d{1,2})(?!\d)"), r"\1.\2"),
    (_re.compile(r"(\d)(?:V|v)/(?=\d)"), r"\1 V / "),
)


def clean(text: str) -> str:
    for rx, rep in _FIXES:
        text = rx.sub(rep, text)
    return text


# Most recognised lines are marketing copy ("HAND GROWN", "SINCE 1933").  The
# attribute we want is always numeric, so keeping only lines that carry a digit
# or a unit word compresses the OCR channel enough to sit alongside the title
# inside one encoder window.
_SALIENT = _re.compile(r"\d|\b(?:oz|ounce|ounces|lb|lbs|pound|pounds|g|gr|gram|grams|"
                       r"kg|ml|l|liter|litre|fl|floz|count|ct|pack|pk|net|wt|weight|"
                       r"mm|cm|m|in|inch|inches|ft|foot|feet|v|volt|volts|w|watt|watts|"
                       r"depth|width|height|deep|wide|tall|high|size|dimension|dimensions)\b",
                       _re.I)


def salient(text: str, max_chars: int = 400) -> str:
    """Quantity-bearing OCR lines only, capped."""
    lines = [ln.strip() for ln in text.split("\n") if _SALIENT.search(ln)]
    return "\n".join(lines)[:max_chars]
