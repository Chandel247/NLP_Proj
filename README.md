# Multimodal Product Attribute Extraction

**Target task: Amazon ML Challenge 2024** — given a product photo and a wanted
attribute name, return its value. Image-only: the dataset has no catalogue text.

The pipeline was first built against the 2025 grocery dataset (see
`README_2025.md` for those results, which remain reproducible). The methodology
— weak supervision by program search, grouped splits, difficulty stratification,
label-noise measurement — is dataset-independent and ported in a day.

## Results

Final numbers on the **test split**, which no decision was tuned on — every
choice below (resolution, epochs, routing rule, controls) was made on validation.
Metric is `dim`: same dimensional family, same magnitude in base units, 1%
tolerance. Brackets are Wilson 95% intervals.

```
                                    n   cascade only          full system
depth / width / height          1,256   38.8% [36.1, 41.5]    58.8% [56.1, 61.5]
the other five attributes       1,013   68.6% [65.7, 71.4]    68.6%  (cascade answers)
ALL eight attributes            2,269   52.1% [50.0, 54.1]    63.2% [61.2, 65.2]
```

The full system is an OCR + tagger + rules cascade for every attribute, with a
Qwen2-VL-2B LoRA fine-tune choosing among OCR's candidates for depth, width and
height, and the cascade as its fallback. All models are local and open; the
fine-tune ran on one free Kaggle T4.

The one-line story: **half the corpus is under-determined by its text.** A
photo of a box reads `12 x 8 x 3 cm`, OCR gets every number, and nothing in the
text says which one is the width. Rules and a tagger get 38.8% of those rows; a
2B VLM taught to point at the right candidate gets 56.0%; and replacing its
photograph with a blank page removes most of that gain, so the model is reading
the image, not a text prior.

The sections below are in the order the work happened, including the calls that
turned out wrong.

## 2024 port status

| Component | State |
|---|---|
| `normalize.py` | **extended** — volt, watt, kilowatt, horsepower, ton, microgram, centi/decilitre, cubic inch/foot. All 25 of 2024's unit strings map (`person`, 1 row, correctly rejected) |
| `evaluate.py` | **generalised** — gold is now an interval `(low, high, unit)`; 4.7% of 2024 labels are genuine ranges (`[110.0, 130.0] volt`). Scalars are the degenerate case, so 2025 numbers are unchanged |
| `amazon2024.py` | **new** — 99.5% of golds parse; splits by `group_id` |
| `ocr.py` / `ocr_pass.py` | **extended** — resolves 2024 images, `--dataset 2024` |
| `label_spans.py`, `tagger.py`, `train_s1.py` | **unchanged** — port as-is |
| OCR cache | **complete** — all 24,926 images of the rebuilt 25,000-row working set, plus the 175 gold images |

## The finding that matters for the team

Recoverability — can the gold value be found in the OCR text at all? — measured
on the **identical 175-row gold set** used by the existing pipeline:

```
                                  n   gold found in OCR
Apple Vision (this repo)        175              64.0%
EasyOCR (existing pipeline)     175              42.3%
```

Apple Vision returned text on **175/175** images. Over the full 3,789-row
working set it reaches **69.5%**.

The existing pipeline's walkthrough attributes 57.7% of its errors to
"OCR / candidate-extraction failure" and names image 52930 — a `100W` printed
vertically on a cable connector — as a known, fixable case needing EasyOCR's
`rotation_info`. It needs no fix:

```
EasyOCR      →  "6 1"              (confidences 0.20, 0.08)
Apple Vision →  ILINKER 100W42PD   (gold: 100.0 watt)
```

**Projected impact.** The existing pipeline reaches 32.2% accuracy against a
42.3% recoverability ceiling — 76% of what its OCR makes available. Holding that
efficiency and raising the ceiling to 64.0% projects to roughly **48% accuracy
from the OCR swap alone.** That is a projection, not a measurement, and should
be confirmed by re-running the pipeline on this cache.

Apple Vision is also free, ships with macOS, needs no GPU, and ran at 31 img/s
on 6 workers.

## S0 baseline on 2024

Rules only — no learned model. The first move is dimensional: `wattage` can be
answered only by a power quantity, `item_volume` only by a volume one, which
discards most OCR noise for free since barcodes and batch codes carry no unit.

**Working set (n=3,789).** `dim` is the primary metric here, not `value`: 2024
gold units are normalised by the challenge (gram, kilogram) while packaging
prints what it likes (`12 OZ`), so a dimensionally correct answer in a different
unit is a success that `value` would score as a failure.

```
attribute                            n   cover   value     dim  strict
item_volume                        795   64.4%   50.7%   52.8%   49.8%
item_weight                        753   79.4%   58.3%   63.1%   58.0%
voltage                            751   81.6%   70.3%   70.3%   70.3%
maximum_weight_recommendation      747   71.5%   60.2%   60.8%   60.0%
wattage                            743   78.1%   70.7%   70.7%   70.7%
ALL                              3,789   74.9%   61.9%   63.4%   61.6%
```

Four candidate-selection heuristics were compared (`first`, `largest`,
`smallest`, `cued`). The cue-proximity heuristic won by **+0.2 points** over
simply taking the first candidate — not enough to justify its complexity, and
worth reporting as such.

### Head-to-head on the 175-row gold set

```
                                accuracy   coverage   precision when answered
S0 rules + Apple Vision (this)     59.4%      69.1%                     86.0%
Existing pipeline (EasyOCR +
  DistilBERT + abstention)         32.2%      36.0%                     90.3%
```

Both are scored against the same **provisional** scraped labels, so both inherit
whatever noise those carry. The existing pipeline is more precise when it
commits; it commits half as often. The gap traces almost entirely to
recoverability: 64.0% vs 42.3% on these exact rows.

### The strategic read

```
OCR recoverability ceiling : 70.5%
S0 dim accuracy            : 63.4%
S0 captures 90% of what OCR makes available
```

On the 2025 corpus S0 captured only 75% of its ceiling, which left a learned
model plenty to win. **On 2024 there is ~7 points of headroom above S0 inside
the current OCR**, and 29.5% of rows where OCR finds nothing at all. The lever
here is the image channel, not the selection model — the opposite of 2025.

That is a projection about where effort pays, not a measurement that a tagger
cannot help; the 7 points are real and worth contesting.

## Rebuilt working set — the real task

The earlier 3,808-row sample was stratified ~equally across five attributes.
That distorted everything measured on it. Rebuilt by simple random sample from
the full 263,859 rows, so proportions are the corpus's own:

```
                        old sample        rebuilt
rows                         3,808         25,000
attributes                       5              8
group_ids                      166            712
splits              2823/677/308   19828/2902/2270
```

`depth`, `width` and `height` — **50.3% of the corpus** — had been dropped
entirely. Adding the length units took label parseability from 67.15% to 99.78%.

### S0 and the ceiling, per attribute (n=25,000)

```
attribute                             n   ceiling   S0 dim  captured  med cands
item_weight                       9,733     76.4%    68.3%       89%          1
depth                             4,259     71.9%    33.1%       46%          2
height                            4,168     72.3%    35.7%       49%          3
width                             4,147     71.0%    29.7%       42%          2
voltage                             893     80.9%    75.0%       93%          1
item_volume                         780     61.9%    54.6%       88%          1
wattage                             730     75.8%    71.5%       94%          1
maximum_weight_recommendation       290     70.0%    61.7%       88%          1
ALL                              25,000     73.7%    50.3%       68%
```

### One column, two tasks — again

```
                              share   ceiling    S0     captured
length (depth/width/height)   50.3%     71.7%   32.8%       46%
everything else               49.7%     75.6%   68.0%       90%
```

**The ceilings are nearly identical.** OCR reads dimensions as well as it reads
weights — 71.7% vs 75.6%. The entire difference is *selection*: the median row
offers 2–3 same-family candidates for a length attribute against 1 for
everything else, because `depth`, `width` and `height` share one dimensional
family and a photo reading `12 x 8 x 3 cm` presents all three at once. The
family filter that carries the other five attributes cannot separate them.

### This reverses the earlier strategic call

Measured on the old five-attribute sample, S0 captured 90% of its ceiling, and
the conclusion was "~7 points of headroom above the rules; the lever is the
image channel, not the selection model." That was true *of those five
attributes* — which are half the corpus.

On the real task, **39 points sit in the length attributes as a pure
disambiguation problem**: the answer is in the OCR text 71.7% of the time and
S0 finds it 32.8% of the time. Weighted by their 50.3% share, that is
**~19.6 points of blended accuracy** available to a model that learns which
number is which — precisely what the S1/S3 tagger exists to do.

The lever is the selection model after all. It only looked otherwise because
half the task was missing from the sample.

## The tagger on 2024 — and why length stays unsolved

First attempt failed outright: **34.0% dim against S0's 45.8%**, abstaining on
56% of rows. Diagnosis, not guesswork:

```
length rows diagnosed: 300
  no tags predicted at all       160   53.3%
  answered                        90   30.0%
  SIZE tagged but no UNIT         41   13.7%
```

The model collapsed to predicting all-`O`. With ~44 tokens of which 2-3 carry a
span, `O` is ~93% of targets; under genuine ambiguity (six candidate numbers,
only the attribute name distinguishing them) the cross-entropy optimum is to
predict nothing. The 2025 inputs were 4x longer with far richer context, which
is why the collapse did not appear there.

Two fixes: span tags weighted 8x against `O`, and a decode fallback that uses
the unit head's prediction when a SIZE span is tagged without a UNIT span.

```
                        before    after
tagger abstention        56.0%    13.8%
tagger dim               34.0%    48.4%
cascade dim              46.5%    49.9%
cascade - S0             +0.8     +4.1
```

### Results (val, n=2,902)

```
                              S0    tagger   cascade   ceiling
LENGTH (50.3%)             31.8%     35.9%     36.2%     71.7%
OTHER  (49.7%)             67.7%     68.2%     71.4%     75.6%
ALL                        45.8%     48.4%     49.9%     73.7%
```

**Non-length attributes are essentially solved**: 71.4% against a 75.6% ceiling
is 94% of what OCR makes available. **Length is not**: +4.4 points against a
39-point gap.

### Why a dimension parser will not fix it

The obvious next idea was an `A x B x C` parser assigning by position. Measured
before building, and it does not exist:

```
length rows: 12,574
  neither                                 12,200   97.0%
  has A x B x C triple                       213    1.7%
  has an explicitly labelled dimension       163    1.3%

gold matches which triple position (rows with a triple):
  depth   pos1: 18%  pos2: 28%  pos3: 55%   (n=40)
  width   pos1: 22%  pos2: 26%  pos3: 52%   (n=50)
  height  pos1: 18%  pos2: 14%  pos3: 68%   (n=65)
```

97% of length rows have no triple and no labelled dimension, and where a triple
does exist all three attributes favour position 3 - so there is no positional
convention to exploit either. The parser would reach ~0.9% of the corpus.

### What the length half actually needs

Typical length OCR looks like this:

```
width   "5.7cm/2.24in  9.3cm/3.66in  4cm/1.57in  3.5cm/1.37in  12.3cm/4.84in"
        gold 9.3 centimetre
```

The number is present - that is why the ceiling is 71.7% - but **nothing in the
text says which dimension it is.** The information lives in the image's arrows
and callouts, i.e. in the spatial relationship between an annotation and the
axis it labels. That is not recoverable by any text model, however good.

**This reverses the earlier demotion of the VLM rung.** It was demoted when
measured on the five-attribute sample, where headroom above the rules was ~7
points. On the real task, half the corpus needs exactly what a VLM provides:
associating a printed dimension with the geometry it annotates. The VLM is no
longer upside - for `depth`/`width`/`height` it is the only remaining lever.

## The VLM rounds — a negative result, and what it licenses

Three zero-shot rounds with Qwen2-VL-2B on 398-400 held-out length rows, run on
a Kaggle T4.

```
variant                                 n  coverage     dim          95% CI
round 1 - open-ended, may abstain     400     21.2%    11.0%    [ 8.3, 14.4]
round 2A - forced open-ended          398     76.9%    35.4%    [30.9, 40.2]
round 2B - select from OCR candidates 398     76.9%    34.2%    [29.7, 39.0]
-------------------------------------------------------------------------
S0 rules over OCR                                      31.8%
tagger + rules cascade                                 36.2%
gold among candidates (selection ceiling)              70.1%
```

**Round 1's 11.0% was a framing artefact, not a capability limit.** It answered
only 21.2% of rows because the prompt offered `"unknown"` — yet when it did
commit it was 51.8% precise against the cascade's 40.7%. Removing the escape
hatch moved it 11.0% → 35.4%.

**But no variant beats the text pipeline.** Every interval overlaps the
cascade's 36.2%. Reframing the task as multiple choice over OCR candidates did
not help either (34.2%).

### Is it reasoning, or guessing?

Restricted to the 252 rows where the gold value is *definitely* among the
candidates offered — so the ceiling is 100% by construction:

```
 n_cands  rows     VLM     1/k
       2    82   61.0%   50.0%
       3    74   32.4%   33.3%
       4    39   43.6%   25.0%
       5    57   31.6%   16.9%
   -----------------------------
     all   252   43.3%   33.7%
```

It is **+9.6 points above uniform random overall**, beats chance clearly at 4-5
candidates, and sits *exactly at chance* with 3. And the letters it picks are
badly skewed:

```
(b) 36.9%   (c) 35.4%   (a) 11.1%   (d) 8.1%   (e-h) 8.5%
```

With 3.52 candidates on average, an unbiased selector would spread near 28% each
across (a)-(d). Position (a) gets 11%. **Much of what looks like selection is
positional habit.**

### Conclusion, and why LoRA is now justified rather than hopeful

Zero-shot, a 2B VLM does **not** resolve depth from width. It is weakly above
chance and level with a text-only pipeline, so on present evidence
`depth`/`width`/`height` remain unsolved and should be reported as such.

But the failure mode is specific and learnable:

1. The supervised target is clean — weak supervision already labels **8,912
   length rows** with the correct span, i.e. exactly which candidate is right.
2. The dominant error is **positional prior**, which is the canonical thing
   fine-tuning corrects; the model is falling back on a habit because it has no
   task-specific signal.
3. The headroom is enormous and measured: **43.3% against a 100% ceiling** on
   rows where the answer is provably in the option list.

So a LoRA fine-tune is the next step - not as a hope that a bigger model helps,
but because the diagnosis names a bias that supervision removes.

## S3 — the LoRA fine-tune

The task given to the model is deliberately narrow: here is the photo, here are
the length values OCR read from it as lettered options, answer with the letter
of the requested dimension. It is never asked to read or measure.

```
base            Qwen2-VL-2B-Instruct, vision encoder frozen
adapter         LoRA r=16, alpha=32, dropout 0.05, all attention + MLP projections
                18,464,768 trainable parameters (0.83%)
data            6,036 train-split length rows with 2+ candidates and a correct one
supervision     the answer letter only; prompt and image tokens are masked
augmentation    option order reshuffled on every __getitem__
schedule        1 epoch, batch 1 x accumulation 8 (754 steps), lr 1e-4
images          512 x 512 max pixels (~335 visual tokens)
hardware        one Kaggle T4: 394 min, peak 12.6 GiB, no gradient checkpointing needed
```

Reshuffling is the direct answer to the zero-shot diagnosis: if option order
changes every time a row is seen, a positional habit earns nothing. Evaluation
keeps options in natural OCR order.

The schedule was shaped by the free tier, not chosen freely. Kaggle caps a
session at 12 hours; the first plan (two epochs at 768 x 768) projected 25 hours.
A TPU was ruled out — Qwen2-VL's visual token count varies with aspect ratio,
which forces `torch_xla` to recompile on nearly every batch.

### Results (validation, the same 406 held-out rows as the zero-shot rounds)

```
system                                   n      dim          95% CI
S0 rules                               406    29.1%    [24.9, 33.7]
tagger + rules cascade                 406    32.8%    [28.4, 37.5]
VLM zero-shot, select a letter         398    34.2%    [29.7, 39.0]
VLM + LoRA, select a letter            406    51.5%    [46.6, 56.3]
-------------------------------------------------------------------
gold among candidates (selector ceiling)      70.0%
```

Against the cascade on identical rows, the fine-tune fixes 110 and breaks 34
(exact McNemar p = 1.5e-10). On the 257 rows with 2+ candidates and a correct
one present, it chooses right **70.8%** of the time, against 35.5% for uniform
random (zero-shot scored 43.3% on the comparable 252-row set above).

Its letters now follow the answers:

```
                           (a)     (b)     (c)     (d)   (e-h)
where the gold answer is  36.6%   33.9%   16.0%    4.3%    9.3%
zero-shot picked          11.1%   36.9%   35.4%    8.1%    8.5%
LoRA picked               37.5%   30.7%   19.5%    4.0%    8.3%
```

Before believing it:

- **Rescored** independently of the notebook: 209/406, zero disagreements.
- **Leakage:** no filename or byte-identical image is shared with training. 21
  perceptual-hash near-matches were inspected by eye — different products with
  different gold values; excluding them anyway gives 51.9%.
- **Text prior:** shuffling removes position but not magnitude ("height is the
  biggest number"). The best text-only magnitude rule, fitted on training rows,
  scores 46.3% on those 257 rows. The fine-tune scores 70.8%.

## S4 — does it use the image? The blank-image control

A 2B model could have learned a subtler prior from the candidate text alone. The
control re-ran the same adapter, prompt and processor on the 277 rows with 2+
candidates, changing only the pixels:

- `real` — the product photo (reproduced the training run: 277/277 identical answers)
- `blank` — a white image of the **same size**, so the visual token count is unchanged
- `swapped` — a different product's photo, via a fixed derangement

```
257 rows, 2+ candidates, correct one present
condition        dim         95% CI     vs real
real           70.8%   [65.0, 76.0]
blank          47.9%   [41.8, 54.0]     -79 / +20   p = 1.8e-9
swapped        44.0%   [38.0, 50.1]     -86 / +17   p = 2.8e-12
text-only rule 46.3%
random         35.5%
```

The model did learn a text prior — worth roughly what the best text rule gets —
and **the photograph adds 22.9 points on top of it.** Blank vs swapped is not
significant (p = 0.34). Split by attribute (n = 84-87 each, so about ±10 points):

```
            real    blank   swapped   photo is worth
depth      75.9%    66.7%     55.2%        +9.2
width      65.5%    35.7%     35.7%       +29.8
height     70.9%    40.7%     40.7%       +30.2
```

Depth is mostly a text problem — it is often the largest number on a size chart.
Width and height are not: without the photo the model is near chance on both.

## S5 — routing between the cascade and the VLM

The two systems fail in different places. Because the 406 validation rows were
the only evaluation set, the routing rule was **fixed before it was scored**:
take the VLM's answer whenever it gives one; fall back to the cascade when OCR
produced no length candidate for the VLM to choose from.

```
validation, 406 rows           dim          95% CI
cascade (S2)                 32.8%    [28.4, 37.5]
VLM (S3)                     51.5%    [46.6, 56.3]
router (S5)                  54.9%    [50.1, 59.7]    +14 / -0 vs VLM, p = 1.2e-4
oracle (best of both)        59.9%    [55.0, 64.5]
```

The gain is mechanical: the VLM is silent on the 94 rows with no candidates, and
the cascade's own span extraction is right on 14 of them.

A learned router was considered and not built. The remaining 20 rows to the
oracle are all disagreements where both answered, and on the 179 disagreements
the VLM is right 110 times to the cascade's 20. In 17 of those 20, the cascade's
answer was already one of the VLM's options, so "the cascade saw a number the
VLM didn't" does not separate them. Isolating 20 rows without losing any of 110
would need the VLM's confidence — worth at most 4.9 points, and not measurable
honestly on rows already used for every other decision.

## S6 — the test split

Run once, with nothing changed, on every length row of the test split. One row
was removed first: splits are by `group_id`, but `41Ol0KtcHdL.jpg` is listed under
group 347320 (train, depth) and 558806 (test, height), and the model had trained
on that exact photo. No other test image appears in the training package.

```
test, 1,256 length rows        dim          95% CI       validation
cascade (S2)                 38.8%    [36.1, 41.5]      32.8%
VLM (S3)                     56.0%    [53.2, 58.7]      51.5%     +298 / -82 vs S2, p = 6.5e-30
router (S5)                  58.8%    [56.1, 61.5]      54.9%     +36 / -0 vs S3,   p = 2.9e-11
oracle                       62.5%    [59.8, 65.1]      59.9%

by attribute     n   cascade     VLM    router
depth          358     35.5%   52.8%     56.7%
width          431     31.8%   50.8%     53.6%
height         467     47.8%   63.2%     65.3%
```

**Every system scores higher on test, including the text-only cascade (+6.0
points).** The test rows are easier, so the absolute figures flatter; what
carries over is the gaps, and they held: VLM over cascade +17.2 (validation
+18.7), router over VLM +2.8 (validation +3.4).

On rows with 2+ candidates and a correct one present the VLM picks right 75.3%
of the time (753 rows); the selector ceiling is 70.8% (validation 70.0%). The run
also saved the model's probability over the option letters for all 811 rows it
answered — its argmax matches the greedy answer on 811/811 — so a confidence
router can later be fitted on validation and checked here without another GPU run.

Whole task, adding the five non-length attributes, where the cascade answers alone:

```
attribute                          n   cascade
item_weight                      873     69.3%
maximum_weight_recommendation     50     64.0%
wattage                           38     52.6%
voltage                           30     66.7%
item_volume                       22     81.8%
ALL eight, cascade only        2,269     52.1%   [50.0, 54.1]
ALL eight, cascade + router    2,269     63.2%   [61.2, 65.2]
```

The four small attributes have 22-50 test rows each; read their individual
figures to ±15 points or worse.

## What went wrong along the way

Recorded because each was stated confidently and then measured:

1. **"The multimodal arm is worth about ten points."** Measured on the 2025 corpus: 4.5.
2. **"Label noise is 1.93%."** Measured on a sample that over-weighted one broken
   attribute tenfold. At natural frequency: 0.188%.
3. **"The rules capture 90% of the ceiling, so the lever is the image channel,
   not selection."** True of five attributes that were half the corpus; on the
   full task the rules capture 46% of the length ceiling.
4. **"A dimension-triple parser will close the length gap."** 97% of length rows
   have no triple; it would reach 0.9% of the corpus.
5. **"The bar for the VLM is the cascade's 36.2%."** That was the whole validation
   split. On the VLM's 406 rows the cascade scores 32.8%.
6. **"If fine-tuning works, the letters will flatten toward a quarter each."**
   Gold itself is skewed in OCR order; a correct model matches the skew.
7. **"The 17 rows the cascade finds outside the candidate list are most of the
   oracle's headroom."** They are half of it, and routing already recovers 14.

## Limitations

- **Labels are the challenge's own scraped values.** 13 of the 175 gold-set rows
  were proved wrong automatically; the rest are unverified. A review sheet with 65
  decision-relevant rows triaged first exists, but hand verification has not been
  done. Every figure here inherits that noise.
- **Grouped splits are a partial guard.** `group_id` is category-level (712 groups
  over 25,000 rows), and at least one photo crosses groups (see S6).
- **Validation reuse.** The 406 validation rows drove every decision from S3 to S5,
  which is why S6 exists; treat validation figures as optimistic.
- **The VLM only chooses.** 29.2% of test length rows have no correct candidate in
  OCR, and a selector cannot reach them. OCR recall is the next ceiling.
- **Apple Vision is macOS-only.** The OCR cache transfers anywhere; the engine
  does not. A cross-platform backend (RapidOCR, PaddleOCR) has not been benchmarked.
- **No abstention for inapplicable attributes.** The system answers whenever it
  finds a same-family number; there is no `attribute_not_applicable` detector.
- **The training log prints loss 8x too high** (a leftover accumulation factor in
  the print only); optimisation was unaffected.

## Future work

1. Hand-verify the 65 triaged gold rows — the one step that cannot be automated.
2. A confidence router: fit a threshold on validation VLM probabilities (one
   inference run, ~8 min), check it on the saved test probabilities. Upper bound
   about 3.7 points on test length rows.
3. OCR recall for the 29% of length rows with no correct candidate: tiling,
   upscaling and rotation sweeps before any larger model.
4. Extend the selector to `item_weight`, which is 39% of the corpus at 69.3%.

## Reproducing

Local (M2 Pro, MPS):

```bash
python tests/test_normalize.py && python tests/test_spans.py
python src/build_workingset.py                       # 25,000-row working set, grouped splits
python src/download_images.py                        # ~34 GB of product photos
python src/ocr_pass.py --dataset ws2024 --csv data2024/working_set.csv --out cache/ocr_ws2024.jsonl
python src/make_spans_2024.py                        # weak supervision by program search
python src/train_2024.py                             # tagger + cascade (S1, S2), val scores
python src/build_lora_package.py                     # -> kaggle/vlm_lora_data.tar.gz
python src/build_test_package.py                     # -> kaggle/vlm_test_data.tar.gz
```

Kaggle (T4, Internet on; attach the package as a dataset):

```
kaggle/vlm_lora_kaggle.ipynb                 S3  train + validation eval   ~6.6 h
kaggle/control/vlm_control_kaggle.ipynb      S4  real / blank / swapped    ~23 min
kaggle/test/vlm_test_kaggle.ipynb            S6  test split, once          ~25 min
```

Scoring, from the downloaded outputs:

```bash
python src/route_2024.py                     # S5 on validation
python src/score_test_2024.py                # S6: test split, length and whole task
```

The written report with charts is `report/plan.html`.

## Live demo

A local web app that runs the full system on one photo and attribute, showing
each stage as it finishes: OCR boxes over the photo, the same-family
candidates, the tagger's spans, the VLM's probability for each option, and the
router's decision. A **blank-image control** button re-runs the S4 control live.

```bash
cd demo_web && npm install && npm run build && cd ..   # once
python src/demo/make_examples.py      # pick test-split examples, record them for replay (~5 min)
python src/demo/server.py             # http://127.0.0.1:8000  (~35 s to load both models)
python src/demo/server.py --replay    # fallback: streams the recorded runs, loads no models
python src/demo/parity.py             # check the demo reproduces the S5 validation numbers
```

`src/demo/pipeline.py` reuses the evaluated code, checkpoints, prompt and image
preprocessing unchanged; it merges the LoRA adapter (`kaggle/qwen2vl-dim-lora`)
into Qwen2-VL-2B and runs on MPS at about 1 s per VLM query. `POST /api/extract`
streams one JSON event per stage (NDJSON). For front-end development, run the
server and `npm run dev` in `demo_web/`; Vite proxies `/api` to port 8000.
