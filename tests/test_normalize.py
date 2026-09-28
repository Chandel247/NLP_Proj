"""Unit tests for the taxonomy.  Run: python -m pytest tests -q  (or this file directly)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import normalize as N

def check(cond, msg):
    if not cond:
        raise AssertionError(msg)

def test_unit_aliases():
    for raw, want in [
        ("Ounce", "ounce"), ("ounce", "ounce"), ("oz", "ounce"), ("OZ", "ounce"),
        ("ounces", "ounce"), ("Fl Oz", "fluid_ounce"), ("FL OZ", "fluid_ounce"),
        ("Fl. Oz", "fluid_ounce"), ("fl.oz", "fluid_ounce"),
        ("Fluid Ounces", "fluid_ounce"), ("fluid ounce(s)", "fluid_ounce"),
        ("Count", "count"), ("ct", "count"), ("COUNT", "count"), ("Cou", "count"),
        ("pound", "pound"), ("LB", "pound"), ("lbs", "pound"),
        ("gramm", "gram"), ("Grams(gm)", "gram"), ("g", "gram"), ("kg", "kilogram"),
        ("millilitre", "milliliter"), ("mililitro", "milliliter"), ("ml", "milliliter"),
        ("Liters", "liter"), ("ltr", "liter"), ("Gallon", "gallon"),
        # container nouns collapse to count
        ("Jar", "count"), ("Bottle", "count"), ("Pouch", "count"), ("K-Cups", "count"),
        ("Tea Bags", "count"), ("SACHET", "count"), ("BAG", "count"),
        # junk must be rejected, not guessed
        ("None", None), ("NA", None), ("-", None), ("---", None), ("....", None),
        ("1", None), ("12.54", None), ("product_weight", None),
        # compound: last recognisable token wins
        ("Count / Count", "count"), ("BOX/12", "count"), ("16 ounces", "ounce"),
        ("7,2 oz", "ounce"), ("1.76 Ounce", "ounce"),
    ]:
        check(N.normalize_unit(raw) == want, f"normalize_unit({raw!r}) -> {N.normalize_unit(raw)!r}, want {want!r}")

def test_conversions():
    check(abs(N.convert(1, "pound", "ounce") - 16) < 1e-9, "1 lb != 16 oz")
    check(abs(N.convert(5, "pound", "ounce") - 80) < 1e-9, "5 lb != 80 oz")
    check(abs(N.convert(1, "liter", "milliliter") - 1000) < 1e-9, "1 L != 1000 ml")
    check(abs(N.convert(100, "milliliter", "fluid_ounce") - 3.3814) < 1e-3, "100 ml != 3.38 fl oz")
    check(abs(N.convert(1, "kilogram", "gram") - 1000) < 1e-9, "1 kg != 1000 g")
    check(N.convert(1, "ounce", "fluid_ounce") is None, "mass->volume must be refused")
    check(N.convert(1, "count", "gram") is None, "count->mass must be refused")

def test_find_quantities():
    q = N.find_quantities("La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)")
    check([(x.value, x.unit) for x in q] == [(12.0, "ounce")], f"got {q}")
    q = N.find_quantities("Albanese Gummi Bears, Sugar Free, 5-Pound Bags (Pack of 2)")
    check([(x.value, x.unit) for x in q] == [(5.0, "pound")], f"got {q}")
    q = N.find_quantities("Rani Mango Chutney 10.5oz (300g) Glass Jar")
    check([(x.value, x.unit) for x in q] == [(10.5, "ounce"), (300.0, "gram")], f"got {q}")
    q = N.find_quantities("Colour Mill Food Coloring, 100 Milliliters (Navy)")
    check([(x.value, x.unit) for x in q] == [(100.0, "milliliter")], f"got {q}")
    # bare container nouns must NOT be read as units from prose
    check(N.find_quantities("comes in a 3 jar gift set") == [], "container noun leaked into text units")
    # a decimal must not be split into two integers
    check([x.value for x in N.find_quantities("net 16.9 fl oz")] == [16.9], "decimal split")

def test_pack_count():
    for text, want in [
        ("12 Ounce (Pack of 6)", 6), ("8 Ounce (Pack of 4)", 4),
        ("Pretzels, Pack of 6", 6), ("6-Pack of soda", 6), ("6 Pack", 6),
        ("Sardines 4.25 oz - 3PC", 3), ("Set of 4 mugs", 4),
        ("Case of 12", 12), ("6 x 12 oz cans", 6),
        ("plain item with no pack", None),
        ("Est. 1925 heritage brand", None),   # year must not become a pack count
    ]:
        check(N.find_pack_count(text) == want, f"find_pack_count({text!r}) -> {N.find_pack_count(text)!r}, want {want!r}")

if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"  PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"  FAIL {name}: {e}")
    print("\nall passed" if not fails else f"\n{fails} test(s) failed")
    sys.exit(1 if fails else 0)
