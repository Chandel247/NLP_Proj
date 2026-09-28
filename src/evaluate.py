"""Metrics, and the difficulty stratification that carries the whole report.

Three accuracies, because "correct" is genuinely ambiguous here:

  value  - magnitude within 1%, unit ignored.  The catalogue uses "Ounce" for
           both mass and fluid ounces, so unit disagreement is often a labelling
           convention rather than a model error.
  dim    - same dimensional family AND the same magnitude in base units.  Credits
           a model that answers 1 pound where the gold says 16 ounce.
  strict - identical canonical unit AND magnitude within 1%.

Rows are also bucketed by how the answer relates to the text, which is what
shows *where* each modality pays off instead of one aggregate number.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import normalize as N
import parse

TOL = 0.01
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

BUCKETS = ["literal_name", "arith_name", "convert_name", "elsewhere_text", "text_unrecoverable"]


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return b != 0 and abs(a - b) / abs(b) <= tol


def _in_range(x: float, lo: float, hi: float, tol: float = TOL) -> bool:
    """Gold is an interval.  4.7% of 2024 labels are genuine ranges
    ("[110.0, 130.0] volt" for a universal power supply); a scalar is the
    degenerate case lo == hi, which reduces this to the 1% tolerance.
    """
    return lo * (1 - tol) <= x <= hi * (1 + tol) if lo > 0 else abs(x - lo) <= tol * max(hi, 1)


def _numbers(text: str) -> list[float]:
    return [float(m.group().replace(",", "")) for m in _NUM_RE.finditer(text)]


def gold_of(rec) -> tuple[float, float, str] | None:
    """Gold as (low, high, canonical unit); low == high for a scalar label."""
    if hasattr(rec, "gold"):          # 2024 records carry their own interval
        return rec.gold
    v = parse.to_float(rec.raw_value)
    u = N.normalize_unit(rec.raw_unit)
    return None if v is None or u is None or v <= 0 else (v, v, u)


# --- correctness ----------------------------------------------------------
def score(pred, gold: tuple[float, float, str]) -> dict[str, bool]:
    lo, hi, gu = gold
    if pred.abstained:
        return {"value": False, "dim": False, "strict": False}
    pv, pu = pred.value, pred.unit
    value = _in_range(pv, lo, hi)
    same_family = N.family(pu) == N.family(gu)
    dim = bool(same_family and _in_range(N.to_base(pv, pu),
                                         N.to_base(lo, gu), N.to_base(hi, gu)))
    return {"value": value, "dim": dim, "strict": bool(pu == gu and value)}


# --- difficulty stratification -------------------------------------------
def bucket(rec, gold: tuple[float, float, str]) -> str:
    lo, hi, gu = gold
    gv = lo
    name = rec.item_name
    body = "\n".join(rec.bullets + [rec.description])

    def tiers(text: str) -> str | None:
        nums = _numbers(text)
        if any(_in_range(x, lo, hi) for x in nums):
            return "literal"
        if any(_in_range(a * b, lo, hi) for a in nums for b in nums):
            return "arith"
        # Conversion: a quantity of the right family, optionally multiplied by a
        # pack count or any other number present.
        mults = [1.0] + [float(n) for n in nums if 1 <= n <= 500]
        pack = N.find_pack_count(text)
        if pack:
            mults.append(float(pack))
        for q in N.find_quantities(text):
            if N.family(q.unit) != N.family(gu):
                continue
            conv = N.convert(q.value, q.unit, gu)
            if conv is not None and any(_in_range(conv * m, lo, hi) for m in mults):
                return "convert"
        return None

    t = tiers(name)
    if t == "literal":
        return "literal_name"
    if t == "arith":
        return "arith_name"
    if t == "convert":
        return "convert_name"
    return "elsewhere_text" if tiers(body) else "text_unrecoverable"


# --- reporting ------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.n = 0
        self.abstain = 0
        self.hits: Counter = Counter()
        self.by_bucket: defaultdict[str, Counter] = defaultdict(Counter)
        self.by_rule: Counter = Counter()
        # Net content (mass/volume) and pack count are different tasks sharing
        # one column: weak supervision explains 97%+ of mass/volume rows but
        # only 34% of count rows, because a count is often a bare integer with
        # no unit word attached.  A blended score hides both.
        self.by_family: defaultdict[str, Counter] = defaultdict(Counter)

    def add(self, rec, pred, gold) -> None:
        b = bucket(rec, gold)
        s = score(pred, gold)
        fam = N.family(gold[2]) or "other"
        fam = "mass/volume" if fam in ("mass", "volume") else fam
        self.by_family[fam]["n"] += 1
        self.by_family[fam]["abstain"] += pred.abstained
        for k, v in s.items():
            self.by_family[fam][k] += v
        self.n += 1
        self.abstain += pred.abstained
        self.by_bucket[b]["n"] += 1
        self.by_bucket[b]["abstain"] += pred.abstained
        self.by_rule[pred.rule] += 1
        for k, v in s.items():
            self.hits[k] += v
            self.by_bucket[b][k] += v

    def table(self, title: str = "") -> str:
        L = [f"\n{title}  (n={self.n:,})", "=" * 78]
        cov = 100 * (1 - self.abstain / max(self.n, 1))
        L.append(f"coverage (non-abstain) : {cov:5.1f}%")
        for k in ("value", "dim", "strict"):
            L.append(f"accuracy [{k:6s}]        : {100*self.hits[k]/max(self.n,1):5.1f}%")
        # `cover` and `on-ans` separate the two ways a system can lose a row:
        # answering wrongly, and declining to answer at all.
        L += ["", f"{'difficulty bucket':<22}{'n':>8}{'share':>8}{'cover':>8}"
                  f"{'value':>8}{'strict':>8}{'on-ans':>8}", "-" * 78]
        for b in BUCKETS:
            c = self.by_bucket.get(b)
            if not c:
                continue
            n = c["n"]
            ans = n - c["abstain"]
            L.append(f"{b:<22}{n:>8,}{100*n/self.n:>7.1f}%{100*ans/n:>7.1f}%"
                     f"{100*c['value']/n:>7.1f}%{100*c['strict']/n:>7.1f}%"
                     f"{(100*c['value']/ans if ans else 0):>7.1f}%")
        L += ["", f"{'gold dimensional family':<22}{'n':>8}{'share':>8}{'cover':>8}"
                  f"{'value':>8}{'strict':>8}{'on-ans':>8}", "-" * 78]
        for f, c in sorted(self.by_family.items(), key=lambda kv: -kv[1]["n"]):
            n = c["n"]
            ans = n - c["abstain"]
            L.append(f"{f:<22}{n:>8,}{100*n/self.n:>7.1f}%{100*ans/n:>7.1f}%"
                     f"{100*c['value']/n:>7.1f}%{100*c['strict']/n:>7.1f}%"
                     f"{(100*c['value']/ans if ans else 0):>7.1f}%")
        L += ["", "rule firing counts:"]
        for r, c in self.by_rule.most_common():
            L.append(f"  {r:<26}{c:>8,}{100*c/self.n:>7.1f}%")
        return "\n".join(L)
