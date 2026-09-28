# Multimodal Product Attribute Extraction

Extract structured attributes from e-commerce listings using **catalogue text +
product image**. NLP semester project.

## The idea

`catalog_content` embeds `Value:` and `Unit:` lines. Strip them from the input
and they become **150,000 free, provider-verified gold labels** (75k train +
75k test) for the attribute *net content*. The task is then:

> Given item name, bullets, description and the product photo,
> recover the total net quantity and its unit.

This is not span copying. Measured over all 75k train rows:

| How the answer relates to the text | Share | Skill required |
|---|---:|---|
| Appears verbatim in the item name | 45.7% | span extraction |
| Product of two numbers in the name | 26.2% | extraction + **arithmetic** |
| Needs **unit conversion** (5-Pound → 80 Ounce) | 8.4% | extraction + **normalisation** |
| Only in bullets/description | 3.1% | long-context extraction |
| **Not recoverable from text at all** | **16.7%** | **must read the package image** |

That last row is the empirical justification for going multimodal: a text-only
model cannot exceed ~83% on this corpus. Examples — the size exists only on the
packaging:

```
Organic Vinegar; Apple Cider              → 102.0 fl oz
Snyder's of Hanover Pretzel variety pack  →  21.0 ounce
```

## Results so far

**S0 — rule baseline** (regex + unit normaliser + pack arithmetic), full 75k:

```
coverage (non-abstain)  92.3%
accuracy [value ]       62.5%     magnitude within 1%, unit ignored
accuracy [dim   ]       55.8%     same family, same magnitude in base units
accuracy [strict]       55.5%     identical canonical unit and magnitude

bucket                       n   share    value     dim  strict
literal_name            33,819   45.7%    72.4%    65.1%    64.6%
arith_name              19,367   26.2%    78.5%    67.4%    67.3%
convert_name             6,189    8.4%    92.3%    93.0%    92.3%
elsewhere_text           2,285    3.1%    24.1%    19.6%    19.4%
text_unrecoverable      12,352   16.7%     2.6%     0.4%     0.4%
```

Two findings already worth reporting:

1. **`literal_name` is only 72.4% correct even though the answer is literally in
   the title.** The rules pick the wrong number. Choosing *which* number is the
   size is a learned-model problem — this is the gap S1/S2 must close.
2. **`text_unrecoverable` scores 2.6%, i.e. chance.** Nothing textual will move
   it. This is where OCR and vision must earn their place.

**Unit normaliser:** 135 distinct raw `Unit` strings → 12 canonical units,
**98.68% coverage** over 150k rows. The rejected 1.32% is genuine junk
(`None`, `---`, `product_weight`, `12.54`).

## S1 — weak supervision + a multi-task tagger

We have no span annotations, only the total. But the total is a *program* over
the text — pick a size, pick its unit, optionally multiply by a pack count — so
`label_spans.py` searches for the assignment that reproduces the gold value and
takes that as supervision, preferring the simplest explanation.

```
weak-supervision coverage: 50,942/74,012 = 68.8%

bucket                  labelled       of    rate
literal_name              26,044   33,819   77.0%
arith_name                17,557   19,367   90.7%
convert_name               6,187    6,189  100.0%
elsewhere_text               816    2,285   35.7%
text_unrecoverable           338   12,352    2.7%

tier 0  size alone                32,150   63.1%
tier 1  size x explicit pack      17,106   33.6%
tier 2  size x bare integer        1,692    3.3%
```

The `text_unrecoverable` rate of 2.7% is the point: no span explains those
answers, so the search correctly declines to invent one.

**The ounce bridge.** The catalogue writes `Ounce` in titles for fluid ounces as
often as for mass — `12 Ounce (Pack of 6)` carries a gold label of `72.0 Fl Oz`.
Strict dimensional analysis rejects every such row. `normalize.catalog_convert`
permits exactly four crossings (ounce↔fluid_ounce, pound↔fluid_ounce) and
reports when one fires: **10.5% of labelled rows**. Adding it moved coverage
61.9% → 68.8% and `literal_name` 69.8% → 77.0%.

**Architecture.** Two heads on one `distilroberta-base` encoder:

- *token head* — BIO over SIZE / UNIT / PACK. Learns **which** number is the
  size, the exact thing S0 gets wrong 27.6% of the time inside `literal_name`.
- *unit head* — the canonical gold unit for the listing. No span can tell ounce
  from fluid ounce; the product category can.

Decoding stays deterministic — read the spans, apply pack arithmetic, convert
into the predicted unit. The network chooses; arithmetic is never left to it.
A round-trip test confirms gold spans decode back to the gold value **400/400**,
so any S1 error is a model error, not a pipeline artifact.

**Headroom for the unit head**, measured on val:

```
S0 value-correct                63.8%
S0 strict                       56.5%
S0 strict with a UNIT ORACLE    63.8%     <- ceiling for the unit head: +7.3 pts
```

91% of S0's unit errors (486 of 534 val rows) are the single confusion
`ounce → fluid_ounce`. That one distinction is the unit head's whole job.

### S1 results (val, n=7,307)

```
                     coverage   value     dim  strict
S0 rule baseline        92.7%   63.8%   56.7%   56.5%
S1 multi-task tagger    82.7%   63.6%   58.5%   58.2%
S1+S0 cascade           93.1%   66.1%   60.7%   60.4%      <- +2.3 value, +3.9 strict
```

Read the raw S1 row and the tagger looks like a failure: `value` went *down*
0.3 points. The `on-answered` column says otherwise.

```
                        S0 coverage / on-ans     S1 coverage / on-ans
literal_name                95.4%  /  76.2%         87.8%  /  79.8%
arith_name                  99.9%  /  79.1%         92.4%  /  92.7%
convert_name               100.0%  /  94.2%         96.3%  /  93.8%
elsewhere_text              89.4%  /  29.6%         54.2%  /  26.8%
```

S1 is markedly *more accurate when it commits* — decisively so on `arith_name`,
**92.7% vs 79.1%** — but it answers only 82.7% of the time against S0's 92.7%.
The cause is a training/evaluation mismatch: the tagger only ever saw the 68.8%
of rows for which weak supervision found an assignment, so it never learned to
respond on the rest and defaults to all-`O`. Falling back to the rules exactly
where S1 declines recovers the loss and keeps the gain.

**Where the gains actually came from.** The prediction was that S1 would fix
`literal_name` (S0 picks the wrong number there). It did, but only modestly:
76.2% → 79.8% on answered rows. The large win was `arith_name` at +13.6 points —
deciding *which* number is the pack multiplier is what learning buys, more than
deciding which is the size.

**The unit head underdelivered.** Since `strict` with a perfect unit oracle
equals `value`, the cost of unit errors is `value − strict`: 7.3 points for S0,
5.4 for S1. The head recovered about a quarter of the available headroom, not
the full 7.3.

**Known fix, not yet applied:** include unsolvable rows in training as all-`O`
targets so abstention is learned rather than accidental. That should let S1
stand alone without the cascade.

## The OCR probe — what the 16.7% actually is

Before committing to an OCR pass over 145k images, a 450-image stratified probe
(Apple Vision, 0.14 s/image) asked whether the package image really carries the
answer. `literal_name` is the control: OCR must do well there or the pipeline,
not the hypothesis, is broken.

```
bucket                    n   empty   title     ocr  ocr+fix  title+ocr
text_unrecoverable      300   10.3%    4.3%   16.3%    16.7%      27.7%
literal_name            150    3.3%   81.3%   48.7%    48.7%      82.7%
```

The control works — OCR alone recovers 48.7% of `literal_name` from the image
with no text at all — so the pipeline is sound. On the target band, title+OCR
reaches **27.7%** against a 4.3% baseline.

**Decomposing the 16.7% band** (n=300):

| Outcome | Share of band |
|---|---:|
| Solved by title + OCR | 27.7% |
| Image yields no OCR text at all | 9.7% |
| Gold number on the package, but no program composes it | 7.7% |
| **Gold number appears nowhere — title or package** | **55.0%** |

That last row is not a near-miss effect. Against *any* number in the title or on
the packaging, the best achievable relative error has a **median of 70%**; only
18.2% land within 5%. The labels simply disagree with the product:

```
Benjamins Artifical Rose Water 16 Oz        → gold 1.0 count
Red Kisses 4.16lb                           → gold 60.0 ounce   (4.16 lb = 66.6 oz)
RODELLE Seafood Seasoning, 7.5 Oz           → gold 7.8 ounce
Sweetshop Kit — package reads NET WT 2.5 OZ → gold 2.3281 ounce
```

### Full pass — the probe held

72,287 images in 38.4 min (31 img/s on 6 workers, 25 MB cache). Every cell of
the 300-image probe reproduced on the full 12,352-row band within 1.6 points:

```
outcome                                   probe     full   delta
solved                                    27.7%    28.6%   +0.9
no_ocr_text                                9.7%     8.2%   -1.5
gold_number_present_no_program             7.7%     6.7%   -1.0
gold_number_absent                        55.0%    56.6%   +1.6
```

Weak-supervision coverage, before vs after OCR:

```
bucket                     rows   before    after    gain
literal_name             33,819    77.0%    77.9%   +0.9
arith_name               19,367    90.6%    91.9%   +1.2
convert_name              6,189   100.0%   100.0%   +0.0
elsewhere_text            2,285    35.7%    48.2%  +12.5
text_unrecoverable       12,352     2.7%    28.6%  +25.8
ALL                      74,012    68.8%    74.2%   +5.4
```

**The label-noise floor is confirmed at 9.4% of the corpus** (6,986 rows whose
value appears neither in the text nor on the package) — the probe estimated 9.2%.

### What this means for the project

- **OCR is worth ~3.9 points overall**, not the ~10 I guessed: 16.7% of rows ×
  23.4 points of recovery. Another ~0.6 comes from `literal_name`. Call it
  **~4.5 points of headroom** — still the largest single remaining lever.
- **The label-noise floor is 9.4% of the whole corpus**, measured on all 74,012
  rows. This is a property of a widely-used dataset and a result in its own right.
- **The realistic ceiling is ~87-91%**, not ~83% + everything vision can add.
- **S5 (VLM) is demoted.** After OCR takes its share, what remains in the band is
  the 9.7% no-text plus 7.7% compose-failure — about 2.9 points total, and a VLM
  would capture only part of that. It is no longer the headline system.
- The digit/letter repair (`0Z` → `OZ`) mattered far less than expected: +0.4
  points. Worth keeping, not worth writing home about.

## S3 — the OCR channel

Same architecture as S1, one extra input field: quantity-bearing OCR lines
appended after a `PKG` marker. 43,907 examples (S1 had 40,631), 288-token
window, 50.6 min on MPS.

```
                     coverage   value     dim  strict
S0 rule baseline        92.7%   63.8%   56.7%   56.5%
S1 tagger               82.7%   63.6%   58.5%   58.2%
S1 + S0 cascade         93.1%   66.1%   60.7%   60.4%
S3 tagger (+OCR)        86.0%   64.6%   60.0%   59.8%
S3 + S0 cascade         95.2%   67.1%   62.2%   62.0%   <- +3.2 value, +5.5 strict vs S0
```

**The result that matters** is not the aggregate. It is one bucket:

```
text_unrecoverable      S0      S1   S1+S0      S3   S3+S0
value accuracy         2.9%    2.8%   3.2%   12.7%   13.1%
```

Three systems sat at chance on that band. Adding OCR moved it to 13.1% —
**+9.9 points on the 15.8% of rows that no text-only system could touch.** That
is the multimodal claim of the project, demonstrated rather than asserted.

**Where it cost something.** `arith_name` regressed 87.3% → 84.9% in the
cascade. The likely cause is distractor numbers: packaging is covered in
figures (nutrition panels, barcodes, batch codes) that compete with the real
pack count. Worth confirming in error analysis — it is the clearest argument
for giving OCR its own encoder segment rather than concatenating it as text.

**Headroom vs realisation.** The probe predicted ~4.5 points and weak
supervision did rise +5.4 points as forecast. But model accuracy rose only
~1.0 point over the S1 cascade. The signal is present in the input and the
model is not fully exploiting it — that gap is the honest finding, and the
argument for S4/S2 rather than for more OCR.

## The blended score was hiding two tasks

Error analysis on `literal_name` — 12.7 of the ~25 recoverable points, the
bucket where the answer sits verbatim in the title — split cleanly by the gold
unit's dimensional family:

```
literal_name val rows      n    share   weak-sup solves
mass                   1,844    54.5%            98.7%
volume                   464    13.7%            96.3%
count                  1,076    31.8%            34.3%
```

**94.5% of unexplained `literal_name` rows have gold unit `count`**, and better
extraction patterns recover only 9.1% of them. The reason is structural: a mass
or volume always arrives with a unit word attached (`14 Ounce`), so a program
can find it. A count is usually a bare integer with no marker, and the catalogue
is inconsistent about whether it even wants one:

```
Tofurky - Italian Sausage, 14 Ounce -- 5 per case      → gold  14.0 ounce
Sour Punch - Share Me Bites, 3.5 Ounce -- 144 per case → gold 144.0 count
Durkee Italian Seasoning, 28 Ounce -- 1 each           → gold   1.0 count
```

Identical surface patterns, different targets. Only the gold `Unit` disambiguates
them — which is exactly what the model must predict.

### Results stratified by family (val, n=7,307)

```
                        n   share   cover   value  strict  on-ans
S0          mass/vol 5,524   75.6%   95.9%   77.7%   68.4%   81.0%
            count    1,783   24.4%   82.6%   20.8%   19.4%   25.2%
S3          mass/vol 5,524   75.6%   92.8%   79.3%   73.4%   85.4%
            count    1,783   24.4%   65.0%   19.2%   17.5%   29.5%
S3+S0       mass/vol 5,524   75.6%   97.9%   81.0%   74.8%   82.7%
            count    1,783   24.4%   86.7%   23.9%   22.0%   27.6%
```

The blended 67.1% was concealing both halves. **On net content the system
reaches 81.0% value / 74.8% strict**, close to the practical ceiling. On pack
count it reaches 23.9%, barely above nothing — and that quarter of the corpus
drags the blended figure down by roughly 14 points.

**Recommendation: scope the headline task to net content (mass/volume) and
report pack count separately as a distinct, ill-posed sub-problem.** Pack count
belongs in Track B as its own attribute, where it can have its own label
definition, rather than being silently averaged into net content.

## Layout

```
src/parse.py       catalog_content -> typed Record; model_input() hides targets
src/normalize.py   unit taxonomy, quantity mining, pack arithmetic
src/splits.py      union-find grouped splits (name signature + image)
src/rules.py       S0 baseline / Track-B silver labeller
src/evaluate.py    3 accuracies + difficulty stratification
src/run_baseline.py
src/label_spans.py weak supervision: (value, unit) -> BIO spans
src/make_spans.py  builds cache/spans_{train,val,test}.jsonl
src/tagger.py      S1 multi-task model + deterministic decoder
src/ocr.py         Apple Vision OCR + digit/letter repair
src/probe_ocr.py   feasibility probe over a stratified sample
src/ocr_pass.py    resumable parallel OCR over a whole split
src/analyze_ocr.py before/after comparison
src/train_s1.py    train / evaluate S1 against S0
tests/             normaliser + weak-supervision + round-trip tests
data -> ../project/data   (symlink; 34 GB of images, do not copy)
```

## Quickstart

```bash
python tests/test_normalize.py          # all passing
python src/run_baseline.py              # full corpus, ~6 s
python src/run_baseline.py --split val  # held-out only

python src/make_spans.py                # weak span labels
python src/train_s1.py --epochs 2       # ~30 min on M2 Pro MPS
python src/train_s1.py --eval-only      # S1 vs S0 on val
python tests/test_spans.py              # incl. decoder round trip

python src/ocr_pass.py                  # 72,287 images, ~38 min, 6 workers
python src/analyze_ocr.py               # before/after OCR comparison
```

## Leakage

Catalogues are full of near-duplicates (same product, six pack sizes; two
sellers sharing one photo). Splits are assigned by **group**, not by row —
union-find over a size-stripped name signature and the image file. 75,000 rows
collapse to 65,969 groups, so a random row split would leak ~12% of test.

## Roadmap

| | System | Where |
|---|---|---|
| S0 | rules + normaliser | done |
| S1 | text-only multi-task tagger (distilroberta-base) | done — 34 min on MPS |
| S2 | text-only seq2seq → JSON (Flan-T5-base) | Colab T4 |
| S3 | **+ OCR channel** (Apple Vision, free, no GPU) | done — 38 min OCR + 51 min train |
| S4 | + frozen CLIP/SigLIP image embedding, gated fusion | Colab T4 |
| S5 | LoRA-tuned small VLM (Qwen2-VL-2B / SmolVLM) | Colab T4 |

Prediction worth testing: **S3 > S4**, because the missing signal is *text
printed on the package*, and a CLIP embedding is a poor carrier of "3.38".

**Track B** (open attributes — brand, pack_count, flavour, container_form,
dietary_claims) uses `rules.py` for silver labels over 75k plus ~800
hand-annotated gold items in `gold/` as the only honest eval.

## Data notes

- 75,000 train rows (`price` present) + 75,000 test rows (`price` absent, but
  `Value`/`Unit` present — usable for this task). All images local.
- Corpus is almost entirely grocery: ounce 58.6%, count 24.5%, fl oz 15.2%.
  Do not claim general product coverage.
- Label noise is real: `16-Ounce Bags (Pack of 6)` is labelled `Value=12,
  Unit=Count`. Quantify it on the gold set and report it as a noise floor.
