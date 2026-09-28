"""Unit taxonomy, quantity parsing and pack arithmetic.

The raw `Unit` column holds 135 distinct surface strings over 150k rows -
case variants ("Ounce"/"ounce"/"oz"/"OZ"), foreign spellings ("gramm",
"mililitro", "stueck"), container nouns used as count units ("Jar", "Pouch"),
and outright junk ("---", "product_weight", "12.54").  Everything downstream
compares *canonical* units, never raw strings, so this module is the single
place that decides what a unit means.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --- taxonomy -------------------------------------------------------------
# canonical unit -> (family, multiplier into that family's base unit)
UNITS: dict[str, tuple[str, float]] = {
    "milligram":    ("mass", 0.001),
    "gram":         ("mass", 1.0),
    "kilogram":     ("mass", 1000.0),
    "ounce":        ("mass", 28.349523125),
    "pound":        ("mass", 453.59237),
    "milliliter":   ("volume", 1.0),
    "liter":        ("volume", 1000.0),
    "fluid_ounce":  ("volume", 29.5735295625),
    "cup":          ("volume", 236.5882365),
    "pint":         ("volume", 473.176473),
    "quart":        ("volume", 946.352946),
    "gallon":       ("volume", 3785.411784),
    "tablespoon":   ("volume", 14.78676478),
    "teaspoon":     ("volume", 4.92892159),
    "count":        ("count", 1.0),
    "inch":         ("length", 1.0),
    "foot":         ("length", 12.0),
    # depth/width/height are 50.4% of the full 2024 dataset and are reported
    # mostly in centimetres - without these, a third of every label is
    # unparseable and looks like dataset noise.
    "millimeter":   ("length", 1 / 25.4),
    "centimeter":   ("length", 1 / 2.54),
    "meter":        ("length", 1000 / 25.4),
    "kilometer":    ("length", 1e6 / 25.4),
    "yard":         ("length", 36.0),
    "mile":         ("length", 63360.0),
    "square_foot":  ("area", 1.0),
    # --- 2024 challenge attributes: voltage, wattage, and the wider mass/volume
    # vocabulary its labels use (ton, microgram, decilitre, cubic inch, ...).
    "microgram":    ("mass", 1e-6),
    "ton":          ("mass", 1e6),          # metric ton
    "centiliter":   ("volume", 10.0),
    "deciliter":    ("volume", 100.0),
    "cubic_inch":   ("volume", 16.387064),
    "cubic_foot":   ("volume", 28316.846592),
    "millivolt":    ("voltage", 0.001),
    "volt":         ("voltage", 1.0),
    "kilovolt":     ("voltage", 1000.0),
    "milliwatt":    ("power", 0.001),
    "watt":         ("power", 1.0),
    "kilowatt":     ("power", 1000.0),
    "horsepower":   ("power", 745.6998715823),
    "kilowatt_hour": ("energy", 1.0),
    "candela":      ("luminous", 1.0),
}
BASE = {"mass": "gram", "volume": "milliliter", "count": "count",
        "length": "inch", "area": "square_foot", "voltage": "volt",
        "power": "watt", "energy": "kilowatt_hour", "luminous": "candela"}

# Fluid ounce must be listed before ounce everywhere it is matched, otherwise
# "fl oz" degrades into a mass reading.
_ALIASES: dict[str, tuple[str, ...]] = {
    "fluid_ounce": ("fl oz", "fl. oz.", "fl.oz.", "fl. oz", "fl.oz", "fluid ounce",
                    "fluid ounces", "fluid ounce(s)", "fluid_ounces", "fl ounce",
                    "fl ounces", "floz", "fo"),
    "ounce":       ("ounce", "ounces", "oz", "oz.", "ozs"),
    "pound":       ("pound", "pounds", "lb", "lbs", "lb.", "lbs.", "#"),
    "gram":        ("gram", "grams", "gramm", "gramme", "grammes", "gr", "g",
                    "gm", "gms", "grams(gm)"),
    "kilogram":    ("kilogram", "kilograms", "kg", "kgs", "kilo", "kilos"),
    "milligram":   ("milligram", "milligrams", "mg"),
    "milliliter":  ("milliliter", "milliliters", "millilitre", "millilitres",
                    "mililitro", "millilitro", "ml", "mls", "cc"),
    "liter":       ("liter", "liters", "litre", "litres", "ltr", "ltrs", "l"),
    "gallon":      ("gallon", "gallons", "gal"),
    "quart":       ("quart", "quarts", "qt"),
    "pint":        ("pint", "pints", "pt"),
    "cup":         ("cup", "cups"),
    "tablespoon":  ("tablespoon", "tablespoons", "tbsp"),
    "teaspoon":    ("teaspoon", "teaspoons", "tsp"),
    "foot":        ("foot", "feet", "ft"),
    "microgram":   ("microgram", "micrograms", "mcg", "ug"),
    "millimeter":  ("millimeter", "millimetre", "millimeters", "millimetres", "mm"),
    "centimeter":  ("centimeter", "centimetre", "centimeters", "centimetres", "cm"),
    "meter":       ("meter", "metre", "meters", "metres"),
    "kilometer":   ("kilometer", "kilometre", "kilometers", "kilometres", "km"),
    "yard":        ("yard", "yards", "yd"),
    "mile":        ("mile", "miles"),
    "ton":         ("ton", "tons", "tonne", "tonnes", "metric ton", "metric tons"),
    "centiliter":  ("centiliter", "centilitre", "centiliters", "centilitres", "cl"),
    "deciliter":   ("deciliter", "decilitre", "deciliters", "decilitres", "dl"),
    "cubic_inch":  ("cubic inch", "cubic inches", "cu in", "in3"),
    "cubic_foot":  ("cubic foot", "cubic feet", "cu ft", "ft3"),
    "volt":        ("volt", "volts", "v", "vac", "vdc", "v ac", "v dc"),
    "millivolt":   ("millivolt", "millivolts", "mv"),
    "kilovolt":    ("kilovolt", "kilovolts", "kv"),
    "watt":        ("watt", "watts", "w"),
    "milliwatt":   ("milliwatt", "milliwatts", "mw"),
    "kilowatt":    ("kilowatt", "kilowatts", "kw"),
    "horsepower":  ("horsepower", "hp"),
    "kilowatt_hour": ("kilowatt hour", "kilowatt hours", "kwh", "kw h"),
    "candela":     ("candela", "candelas", "cd"),
    "inch":        ("inch", "inches", "in"),
    "square_foot": ("sq ft", "square foot", "square feet", "sqft"),
    # Container nouns are genuinely count semantics in this catalogue: a Unit of
    # "Jar" with Value 6 means six jars.  Collapsing them into `count` is what
    # lifts canonical coverage past 99%.
    "count":       ("count", "counts", "ct", "cts", "cou", "each", "ea", "piece",
                    "pieces", "pc", "pcs", "unit", "units", "unita", "stuck",
                    "pack", "packs", "pac", "packet", "packets", "sachet",
                    "bottle", "bottles", "jar", "jars", "bag", "bags",
                    "ziplock bags", "box", "boxes", "can", "cans", "carton",
                    "cartons", "tin", "tins", "pouch", "pouches", "bucket",
                    "container", "containers", "capsule", "capsules", "k-cups",
                    "kit", "tea bags", "tea bag", "per package", "per carton",
                    "per box", "paper cupcake liners"),
}
_ALIAS_TO_CANON: dict[str, str] = {a: c for c, al in _ALIASES.items() for a in al}


def _fold(s: str) -> str:
    """Lowercase, strip accents and collapse punctuation/whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[‐-―]", "-", s)
    return re.sub(r"\s+", " ", s)


def normalize_unit(raw: str | None) -> str | None:
    """Map a raw `Unit` string to a canonical unit, or None if unusable."""
    if raw is None:
        return None
    s = _fold(raw)
    if not s or s in {"none", "na", "n/a", "-", "--", "---", "....", "1"}:
        return None
    if s in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[s]
    s2 = s.strip(".").replace("/", " ").strip()
    if s2 in _ALIAS_TO_CANON:
        return _ALIAS_TO_CANON[s2]
    # Compound junk such as "Count / Count", "BOX/12", "16 ounces", "7,2 oz":
    # take the last recognisable unit token in the string.
    for tok in reversed(re.findall(r"[a-z.]+", s2)):
        if tok in _ALIAS_TO_CANON:
            return _ALIAS_TO_CANON[tok]
    return None


def family(unit: str | None) -> str | None:
    return UNITS[unit][0] if unit in UNITS else None


def convert(value: float, src: str, dst: str) -> float | None:
    """Convert within a dimensional family; None across families."""
    if src not in UNITS or dst not in UNITS or family(src) != family(dst):
        return None
    return value * UNITS[src][1] / UNITS[dst][1]


# The catalogue writes "Ounce" in titles for fluid ounces as often as for mass:
# "12 Ounce (Pack of 6)" carries a gold label of 72.0 Fl Oz.  Dimensional
# analysis says mass and volume cannot mix, and for physics that is right - but
# the surface string here is a naming convention, not a measurement.  These are
# the only crossings allowed, and callers are told when one fires so the rate
# can be reported rather than hidden.
_BRIDGE: dict[tuple[str, str], float] = {
    ("ounce", "fluid_ounce"): 1.0,
    ("fluid_ounce", "ounce"): 1.0,
    ("pound", "fluid_ounce"): 16.0,
    ("fluid_ounce", "pound"): 1 / 16,
}


def catalog_convert(value: float, src: str, dst: str) -> tuple[float | None, bool]:
    """convert(), extended by the corpus's ounce/fluid-ounce conflation.

    Returns (converted value, whether the ounce bridge was used).
    """
    v = convert(value, src, dst)
    if v is not None:
        return v, False
    factor = _BRIDGE.get((src, dst))
    return (value * factor, True) if factor is not None else (None, False)


def to_base(value: float, unit: str) -> float | None:
    """Express a quantity in its family's base unit (gram / ml / count)."""
    return None if unit not in UNITS else value * UNITS[unit][1]


# --- quantity extraction from free text -----------------------------------
# Two different jobs, so two different alias sets.  `normalize_unit` reads the
# curated `Unit` column and may safely treat "Jar" as count; free text may not -
# in prose a container noun is almost never a unit ("3 jar gift set" is not a
# quantity), and single letters like "l" collide with sizes.  Only these surface
# forms may attach to a number when mining item names and bullets.
TEXT_ALIASES: frozenset[str] = frozenset((
    "fl oz", "fl. oz.", "fl.oz.", "fl. oz", "fl.oz", "fluid ounce",
    "fluid ounces", "fluid ounce(s)", "fl ounce", "fl ounces", "floz",
    "ounce", "ounces", "oz", "oz.", "ozs",
    "pound", "pounds", "lb", "lbs", "lb.", "lbs.",
    "gram", "grams", "g", "gm", "gms",
    "kilogram", "kilograms", "kg", "kgs",
    "milligram", "milligrams", "mg",
    "milliliter", "milliliters", "millilitre", "millilitres", "ml",
    "liter", "liters", "litre", "litres",
    "gallon", "gallons", "quart", "quarts", "pint", "pints",
    "count", "counts", "ct", "piece", "pieces", "pc", "pcs",
    "millimeter", "millimetre", "mm", "centimeter", "centimetre", "cm",
    "meter", "metre", "inch", "inches", "foot", "feet", "yard", "yards",
    # 2024: these appear on packaging as "100W", "220V", "12 VDC"
    "volt", "volts", "v", "vac", "vdc", "millivolt", "mv", "kilovolt", "kv",
    "watt", "watts", "w", "milliwatt", "kilowatt", "kw", "horsepower", "hp",
    "microgram", "mcg", "ton", "tons", "tonne",
    "centiliter", "centilitre", "cl", "deciliter", "decilitre", "dl",
))

def _alt(aliases) -> str:
    """Longest first so "fl oz" wins over "oz" and "sq ft" over "ft"."""
    return "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_QTY_TEXT_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s*[-\u2010-\u2015]?\s*({_alt(TEXT_ALIASES)})(?![a-z])", re.I
)


# "100-240V", "AC 100~240 V", "3.5 - 9 volt": one unit shared by two numbers.
# Without this only the second number attaches to the unit, so a label of
# [100, 240] volt matches by luck and [110, 130] volt does not match at all.
_RANGE_RE = re.compile(
    rf"(?<![\w.])({_NUM})\s*[-~\u2010-\u2015/]\s*({_NUM})\s*({_alt(TEXT_ALIASES)})(?![a-z])",
    re.I,
)


def find_ranges(text: str, *, source: str = "") -> list[tuple[float, float, str, int, int]]:
    """(low, high, canonical unit, start, end) for shared-unit number ranges."""
    out = []
    for m in _RANGE_RE.finditer(text):
        lo, hi = _num(m.group(1)), _num(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        out.append((lo, hi, _ALIAS_TO_CANON[_fold(m.group(3))], m.start(), m.end()))
    return out


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str          # canonical
    start: int
    end: int
    source: str        # which text field it came from

    def base(self) -> float | None:
        return to_base(self.value, self.unit)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def find_quantities(text: str, *, source: str = "") -> list[Quantity]:
    """All (number, unit) pairs in `text`, using text-safe unit aliases only."""
    return [
        Quantity(_num(m.group(1)), _ALIAS_TO_CANON[_fold(m.group(2))],
                 m.start(), m.end(), source)
        for m in _QTY_TEXT_RE.finditer(text)
    ]


# "(Pack of 6)", "6-Pack", "6 pk", "Set/Case/Box/Bundle of 4", "3PC", "6 x 12 oz"
_PACK_RES = (
    re.compile(r"\((?:pack|set|case|box|bundle|lot)\s+of\s+(\d+)\)", re.I),
    re.compile(r"\b(?:pack|set|case|box|bundle|lot)\s+of\s+(\d+)\b", re.I),
    re.compile(r"\b(\d+)\s*[-\s]?\s*(?:pack|pk|pc|pcs|ct\b|count\b)", re.I),
    re.compile(r"\b(\d+)\s*[x×]\s*(?=\d)", re.I),
)


def find_pack_count(text: str) -> int | None:
    """Multiplier turning a per-unit size into a total.  None when absent."""
    for rx in _PACK_RES:
        m = rx.search(text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 500:      # guard against years, UPCs, weights
                return n
    return None
