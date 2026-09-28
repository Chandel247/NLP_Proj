"""Weak-supervision and decoding tests.

The important one is the round trip: take the gold character spans, project them
onto tokens exactly as training does, then decode those tags back into a value.
If that does not reproduce the gold number, every S1 evaluation number is
meaningless regardless of how well the network fits.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import label_spans as LS
import normalize as N


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_find_assignment():
    # size alone
    a = LS.find_assignment("Judee's Blue Cheese Powder 11.25 oz", 11.25, "ounce")
    check(a is not None and a.tier == 0, f"size-alone failed: {a}")
    # size x explicit pack
    t = "La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)"
    a = LS.find_assignment(t, 72.0, "fluid_ounce")
    check(a is not None, "ounce->fl oz bridge not found")
    check(a.bridged, "bridge flag not set")
    check(t[a.size[0]:a.size[1]] == "12", f"size span {t[a.size[0]:a.size[1]]!r}")
    check(t[a.unit[0]:a.unit[1]] == "Ounce", f"unit span {t[a.unit[0]:a.unit[1]]!r}")
    check(t[a.pack[0]:a.pack[1]] == "6", f"pack span {t[a.pack[0]:a.pack[1]]!r}")
    # unit conversion: 5 lb x 2 = 160 oz
    t = "Albanese Gummi Bears, Sugar Free, 5-Pound Bags (Pack of 2)"
    a = LS.find_assignment(t, 160.0, "ounce")
    check(a is not None and t[a.size[0]:a.size[1]] == "5", f"conversion case: {a}")
    # a decimal size must not be clipped to its first digit
    t = "Bear Creek Soup, 1.9 Ounce (Pack of 6)"
    a = LS.find_assignment(t, 11.4, "ounce")
    check(t[a.size[0]:a.size[1]] == "1.9", f"decimal clipped: {t[a.size[0]:a.size[1]]!r}")
    # nothing in the text explains this label
    check(LS.find_assignment("Organic Vinegar; Apple Cider", 102.0, "fluid_ounce") is None,
          "invented an assignment where the text has no number")


def test_bridge_is_narrow():
    check(N.catalog_convert(1, "ounce", "fluid_ounce") == (1.0, True), "oz->fl oz bridge")
    check(N.catalog_convert(1, "pound", "fluid_ounce") == (16.0, True), "lb->fl oz bridge")
    v, b = N.catalog_convert(1, "pound", "ounce")
    check(abs(v - 16) < 1e-9 and not b, "real conversion must not be flagged as bridged")
    check(N.catalog_convert(1, "gram", "milliliter")[0] is None, "gram->ml must stay refused")
    check(N.catalog_convert(1, "count", "ounce")[0] is None, "count->ounce must stay refused")


def test_encode_decode_round_trip():
    import tagger
    tok = tagger.get_tokenizer()
    rows = tagger.load_jsonl("cache/spans_train.jsonl", limit=400)
    enc = tagger.encode(rows, tok)
    ok = 0
    for i, r in enumerate(rows):
        tags = [max(t, 0) for t in enc["labels"][i].tolist()]   # -100 -> O
        offs = tok(r["text"], truncation=True, max_length=tagger.MAX_LEN,
                   padding="max_length", return_offsets_mapping=True)["offset_mapping"]
        pred = tagger.decode_prediction(r["text"], tags, offs,
                                        tagger.unit_class(r["gold_unit"]))
        if not pred.abstained and abs(pred.value - r["gold_value"]) / r["gold_value"] <= 0.01:
            ok += 1
    rate = ok / len(rows)
    print(f"    round trip: {ok}/{len(rows)} = {100*rate:.1f}%")
    check(rate >= 0.97, f"gold spans decode back to the gold value only {100*rate:.1f}% "
                        f"of the time - decoding is lossy")


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
