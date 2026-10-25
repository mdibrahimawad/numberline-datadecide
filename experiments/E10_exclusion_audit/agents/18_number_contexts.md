# 18 — In what contexts do numbers occur in each training stream? (blind audit)

Auditor aspect: whether the per-recipe number counts c(n) (n = 0..10000, regex `\b\d+\b`, exact 100B-token
stream, seed 2) measure exposure to numbers *as quantities*, or are dominated by non-quantity uses
(years, HTML/Unicode entity codes such as `&#8217;`, code literals, citation identifiers, boilerplate/spam),
which would make counts an invalid proxy. No β or α–β relation was used or looked at; the two β-derived
columns of `count_features.csv` (`old_direct`, `beta_resid`) were dropped on read.

## 0. Data and method (what could and could not be done)

* Inputs: `results/corpus_alpha_datadecide/exact_100b_<recipe>/counts_seed_2.csv` (the exact counts used by
  `src/count_features.py::load_counts`), the non-β columns of `results/datadecide/count_features.csv`,
  and source composition from `results/corpus_alpha_datadecide/<recipe>/mixture.json`.
* **Raw text was not available.** Only per-integer counts are stored in the repo; the token files live on the
  Hugging Face Hub (`allenai/DataDecide-data-recipes`, `docs/datadecide_sampling.md` l.78) and
  `curl https://huggingface.co/...` returned `CONNECT tunnel failed, response 403` (egress policy). So contexts
  are inferred from **count signatures**: excess counts at integers that have a known non-quantity use,
  measured against a local baseline. This identifies the *conspicuous* non-quantity uses only; it cannot
  split the 0–9 / small-number mass (42–45 % of all matches, `F03`) into quantities vs list indices, dates,
  version numbers or decimal fragments. Stated as a limitation, not hidden.
* Scripts (all in scratchpad, run with `/tmp/claude-0/venv/bin/python -I`):
  `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/ctx/contexts.py`,
  `.../ctx/sig2.py`, `.../ctx/spikes.py`, `.../ctx/ratio.py`, `.../ctx/table.py`; outputs `.../ctx/ctx_full.csv`,
  `.../ctx/table.csv`.
* Baselines: local baseline b(n) = running median of log c over n±10 (`median_filter`, size 21) — a single
  spike cannot move a median. Years: log-linear fit on 1700–1899 ∪ 2031–2299, excess over 1900–2030.
  Unicode/entity code points (8192–8303 General Punctuation, 8448–8959 letterlike/arrows/math, 9472–9983
  box/geometric/misc symbols): power-law fit on 6000–8191 extrapolated. Excess = Σ max(c − baseline, 0).
* Signature sets, fixed before looking at per-recipe values:
  - **years** 1900–2030;
  - **entities** the Unicode blocks above (+ `&#39; &#34; &#038; &#160; ©® é` = 39, 34, 38, 160, 169, 174, 233);
  - **code literals** powers of two 16…8192, 255 (RGB), 168/192 (192.168.x.x), ports 443/8080/3306/8443, 768/1366;
  - **CSS hex colours / repdigits** 111…999;
  - **citation identifiers** ISBN prefixes 978/979 and DOI registrant prefixes (10.1016 Elsevier, 10.1007 Springer,
    10.1038 Nature, 10.1002 Wiley, 10.1080, 10.1093, 10.1103, 10.1371, 10.1177, …; 24 numbers);
  - **round numbers** (multiples of 10/100) and 15/25/45/75/95/99/125/365 are treated as **quantity** uses
    (approximate amounts, percentages, prices `x.99`), not as non-quantity.
* Denominator: counts in 10..9999, the range in which R (`F14_n_seen_lt_4000`) is defined and the probe operates.

## 1. Per-model table (all 25 recipes)

Shares in % of counts in 10..9999. `nonq` = years + entities + code + citations + repdigits. `spike_nr` = all
excess at non-round numbers that are >2× their local median (any cause). `c8217/nb` = c(8217)/median of
c(8200–8209, 8225–8234). `R`, `dR` = R and its change when non-quantity spikes are replaced by baselines.
`α_ols_cl` = α_OLS with non-quantity spikes replaced. `b2154 yr` = year excess as % of the probe band
1939–2369 (centre 10^(10/3)); `b10k nonq` = entity excess as % of the 9000–10000 band. code/math = token share
of StarCoder / (Proof-Pile-2 + arXiv) in the recipe pool (mixture.json).

| recipe | family | code% | math% | years | entities | c8217/nb | code lit. | citations | spike_nr | **nonq** | F19 round100 | F23 1–20 share | c0/c1 | R | dR | α_ols | α_ols_cl | b2154 yr | b10k nonq |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dolma1_7 | dolma | 15.38 | 3.11 | 20.65 | 0.08 | 38.41 | 0.6 | 0.1 | 1.74 | **21.54** | 2.48 | 0.49 | 0.91 | 6384 | 41 | -1.47 | -1.451 | 97.4 | 8.31 |
| dolma1_7-no-code | dolma | 0.0 | 3.72 | 22.89 | 0.05 | 42.47 | 0.16 | 0.1 | 1.39 | **23.29** | 2.76 | 0.49 | 0.66 | 7177 | 31 | -1.569 | -1.549 | 98.0 | 11.01 |
| dolma1_7-no-math-code | dolma | 0.0 | 2.07 | 23.74 | 0.06 | 43.91 | 0.14 | 0.1 | 1.43 | **24.13** | 2.78 | 0.48 | 0.65 | 7227 | 31 | -1.572 | -1.552 | 98.02 | 11.6 |
| dolma1_7-no-reddit | dolma | 16.13 | 3.26 | 20.81 | 0.08 | 39.49 | 0.61 | 0.1 | 1.73 | **21.71** | 2.45 | 0.49 | 0.92 | 6279 | 43 | -1.469 | -1.449 | 97.41 | 8.36 |
| dolma1_7-no-flan | dolma | 15.53 | 3.14 | 20.51 | 0.08 | 39.59 | 0.61 | 0.1 | 1.76 | **21.42** | 2.5 | 0.49 | 0.91 | 6426 | 41 | -1.476 | -1.457 | 97.38 | 8.41 |
| dolma1_6plus | dolma | 13.99 | 1.35 | 20.43 | 0.08 | 30.68 | 0.54 | 0.11 | 1.91 | **21.31** | 2.48 | 0.49 | 0.87 | 5707 | 62 | -1.429 | -1.411 | 97.28 | 8.75 |
| c4 | c4 | 0.0 | 0.0 | 24.82 | 0.06 | 31.3 | 0.18 | 0.07 | 2.81 | **25.23** | 2.9 | 0.45 | 0.67 | 7392 | 34 | -1.563 | -1.543 | 98.11 | 12.1 |
| fineweb-pro | fineweb | 0.0 | 0.0 | 25.3 | 0.03 | 28.66 | 0.11 | 0.37 | 1.35 | **25.85** | 2.53 | 0.42 | 0.69 | 7433 | 26 | -1.762 | -1.742 | 97.55 | 10.12 |
| fineweb-edu | fineweb | 0.0 | 0.0 | 26.18 | 0.04 | 39.98 | 0.16 | 0.5 | 1.46 | **26.92** | 2.44 | 0.44 | 0.56 | 7075 | 32 | -1.699 | -1.678 | 97.17 | 8.92 |
| falcon | falcon | 0.0 | 0.0 | 23.76 | 0.05 | 8.87 | 0.13 | 0.11 | 1.39 | **24.15** | 2.64 | 0.48 | 0.61 | 6432 | 52 | -1.505 | -1.484 | 98.05 | 11.47 |
| falcon-and-cc | falcon | 0.0 | 0.0 | 24.06 | 0.06 | 50.74 | 0.13 | 0.11 | 1.42 | **24.46** | 2.78 | 0.47 | 0.63 | 7155 | 34 | -1.581 | -1.561 | 98.08 | 12.34 |
| falcon-and-cc-qc-10p | falcon | 0.0 | 0.0 | 14.48 | 0.08 | 86.46 | 0.32 | 0.07 | 2.79 | **15.05** | 3.46 | 0.54 | 0.65 | 7735 | 25 | -1.679 | -1.665 | 96.68 | 16.04 |
| falcon-and-cc-qc-20p | falcon | 0.0 | 0.0 | 16.42 | 0.07 | 68.55 | 0.24 | 0.11 | 1.87 | **16.93** | 3.21 | 0.52 | 0.62 | 7668 | 26 | -1.679 | -1.664 | 97.01 | 15.1 |
| falcon-and-cc-qc-orig-10p | falcon | 0.0 | 0.0 | 15.75 | 0.08 | 56.03 | 0.39 | 0.08 | 2.0 | **16.41** | 3.09 | 0.54 | 0.61 | 7609 | 28 | -1.648 | -1.633 | 96.62 | 13.32 |
| falcon-and-cc-qc-tulu-10p | falcon | 0.0 | 0.0 | 23.12 | 0.04 | 32.13 | 0.18 | 0.21 | 1.02 | **23.62** | 2.47 | 0.48 | 0.6 | 7030 | 31 | -1.691 | -1.673 | 97.64 | 10.17 |
| dclm-baseline | dclm | 0.0 | 0.0 | 20.78 | 0.08 | 112.19 | 0.22 | 0.22 | 1.47 | **21.37** | 2.62 | 0.53 | 0.49 | 7392 | 32 | -1.596 | -1.578 | 97.66 | 12.74 |
| dclm-baseline-qc-7p-fw2 | dclm | 0.0 | 0.0 | 21.02 | 0.07 | 43.64 | 0.35 | 0.4 | 1.8 | **21.91** | 2.57 | 0.51 | 0.5 | 7194 | 37 | -1.663 | -1.644 | 96.86 | 12.1 |
| dclm-baseline-qc-7p-fw3 | dclm | 0.0 | 0.0 | 21.45 | 0.06 | 12.6 | 0.32 | 0.56 | 2.18 | **22.44** | 2.66 | 0.5 | 0.52 | 7106 | 46 | -1.719 | -1.701 | 96.68 | 14.47 |
| dclm-baseline-qc-fw-3p | dclm | 0.0 | 0.0 | 21.7 | 0.06 | 7.86 | 0.28 | 0.68 | 2.37 | **22.76** | 2.64 | 0.48 | 0.55 | 6999 | 56 | -1.731 | -1.713 | 96.7 | 15.08 |
| dclm-baseline-qc-fw-10p | dclm | 0.0 | 0.0 | 22.39 | 0.05 | 17.61 | 0.26 | 0.52 | 1.85 | **23.29** | 2.59 | 0.48 | 0.52 | 7060 | 44 | -1.723 | -1.704 | 96.84 | 13.27 |
| dclm-baseline-qc-10p | dclm | 0.0 | 0.0 | 15.73 | 0.07 | 87.24 | 0.42 | 0.18 | 1.92 | **16.5** | 2.66 | 0.6 | 0.48 | 7548 | 33 | -1.614 | -1.597 | 97.29 | 11.18 |
| dclm-baseline-qc-20p | dclm | 0.0 | 0.0 | 17.73 | 0.08 | 104.12 | 0.37 | 0.23 | 1.88 | **18.5** | 2.64 | 0.57 | 0.48 | 7487 | 33 | -1.606 | -1.588 | 97.35 | 10.27 |
| dclm-baseline-25p-dolma1.7-75p | mix | 8.46 | 1.86 | 20.56 | 0.08 | 57.1 | 0.43 | 0.15 | 1.58 | **21.31** | 2.52 | 0.5 | 0.77 | 6894 | 36 | -1.513 | -1.493 | 97.47 | 9.43 |
| dclm-baseline-50p-dolma1.7-50p | mix | 4.92 | 0.91 | 20.59 | 0.07 | 76.94 | 0.34 | 0.18 | 1.54 | **21.28** | 2.55 | 0.51 | 0.66 | 7132 | 36 | -1.541 | -1.521 | 97.51 | 10.14 |
| dclm-baseline-75p-dolma1.7-25p | mix | 1.97 | 0.43 | 20.73 | 0.08 | 95.39 | 0.27 | 0.2 | 1.5 | **21.36** | 2.6 | 0.52 | 0.57 | 7324 | 35 | -1.577 | -1.558 | 97.62 | 11.83 |

## 2. What the numbers are, per recipe (characterisation)

**Common to every recipe (not differential):**
* **Years are the largest identifiable non-quantity use.** 64–82 % of all counts in 1000–9999 fall in
  1900–2030 (`F20_year_share_of_d4`, count_features.csv: min 0.644 dclm-baseline-qc-fw-3p, max 0.821
  falcon-and-cc); the *excess* over a smooth power law is 14.5–26.2 % of all counts in 10..9999 (table, `years`).
* **The probe band centred at 2154 (1939–2369) is ~97 % year excess in every recipe** (`b2154 yr`
  96.6–98.1 %). Any count feature evaluated at that probe band measures "years", not quantities near 2000 —
  but uniformly so (spread 1.5 points).
* Round numbers dominate the large single spikes (5000 is 141–524× its neighbours, then 3000, 4000, 6000;
  `ratio.py` output). These are approximate quantities (prices, amounts, distances) and are counted as quantities.
* Quantity-like small spikes 15/25/45/75/95/99 (percentages, quarters, `.99`/`.95` price cents) are 1.0–3.3 %
  of mass (`quantlike_spike_share_mid` in ctx_full.csv).
* HTML/Unicode entity codes are real but tiny in mass: c(8217) is 8–112× its neighbours (the `&#8217;`
  right-quote), and the entity blocks carry 18–32 % of all counts in 8192–9999 (`unicode_excess_share_8192_9999`),
  i.e. 8–16 % of the top probe band 9000–10000, but only 0.03–0.08 % of mass in 10..9999.
* Counting-rule artefacts (definitional, same for all recipes): `1,000` → 1 and 0, `9.99` → 9 and 99,
  `2019-05-12` → 2019, 5, 12, `192.168.1.1` → 192, 168, 1, 1. They inflate 0 and small numbers; c(0)/c(1)
  = 0.48–0.92 (`zero_over_one`).

**Recipe-specific signatures:**
* **Dolma 1.7 family + dolma1_6plus + Dolma-heavy mixes (code present, 14–16 % StarCoder):** clearest
  code-literal signature — c(255) 6.9–8.2× neighbours (RGB/byte max) vs 1.6–1.8× in the same recipes with
  code removed (`c255_over_nb`, ctx_full.csv); 8080 appears among the top spikes (83×) only when code is in;
  pow2 excess `F22` 1.10–1.17 with code vs 0.73–0.82 without; highest c(0)/c(1) (0.87–0.92 vs 0.65–0.66
  without code) — 0 is the commonest code literal. Code literals are nonetheless only 0.5–0.6 % of 10..9999 mass.
* **FineWeb-Edu, FineWeb-Pro and the FineWeb-classifier DCLM filters (qc-7p-fw2, qc-7p-fw3, qc-fw-3p,
  qc-fw-10p): citation/reference contexts.** c(978) (ISBN-13 prefix) is 19.6–48.8× its neighbours in the dclm
  fw filters and 27.5× in fineweb-edu (vs 4–6× in Dolma/C4), c(1016) (DOI 10.1016) 13–27×, plus 1038
  (10.1038) and 8859 (ISO-8859 charset strings) among the top spikes. Citation excess 0.37–0.68 % of mass
  (robust z 2.7–6.6). Also highest sexagesimal (time-of-day) break in fineweb (0.23–0.26) and c4 (0.21).
* **falcon-and-cc-qc-10p / -qc-20p / -qc-orig-10p and dclm-baseline-qc-10p / -qc-20p** (the most selective
  quality filters): the **lowest** year share (years excess 14.5–17.7 % vs 20–26 % elsewhere), the highest
  small-number share (`F23` 0.52–0.60) and the strongest round-number spikes (F19 3.09–3.46 for the falcon
  qc filters; 5000 at 375–524×). Reads as instructional / list / worked-example style text with enumerations
  and round amounts — quantity-rich, not boilerplate. 401 (401(k)) is 8–9× in the dclm qc filters; 911 is
  12–18× in falcon qc and dclm-baseline.
* **Falcon-RefinedWeb:** c(8217)/nb only 8.9 (vs 30–112) — its extraction pipeline decodes entities;
  DCLM-baseline highest (112), i.e. most raw-markup residue — still 0.08 % of mass.
* **C4:** largest share of non-round spikes overall (2.8 %; robust z 2.8; e.g. 365 at 8.7×, 9001 at 74×
  — "ISO 9001"), consistent with C4's commercial/boilerplate web pages; again small in mass.

No recipe shows a templated-spam signature (a single non-round, non-year integer carrying a large share):
the largest single non-round, non-year, non-percent spike is 24 or 12 (hours/months/dozens, "24/7") at 0.16–0.64 % of mass (max dclm-baseline-qc-10p 0.64 %; inline check in `ctx/` scratch run), and the conspicuous identifier spikes (978, 1016, 911, 401, 8217, 8859) are each well below that, and all >20× spikes
together (overwhelmingly round thousands) carry 1.4–8.9 % (`ratio.py`, `mass>20x`).

## 3. Exclusion / flag criteria (fixed a priori) and the models each flags

| # | Rule (threshold) | Justification | Flags | Kind |
|---|---|---|---|---|
| C1 | Identified non-quantity excess (`nonq`) **> 50 %** of counts in 10..9999 | If most counted occurrences are not quantities, c(n) no longer measures exposure to n as a number — the exposure measurement is invalid. 50 % = "dominated". | **none** (max 26.9 % fineweb-edu; min 15.1 % falcon-and-cc-qc-10p) | would be exclusion; not triggered |
| C2 | Removing non-quantity spikes changes the recipe's **rank** in α_OLS or R by **≥ 3** of 25 positions, or changes R by more than 1 between-recipe SD (490) | If the summary statistics used downstream are driven by non-quantity uses, the measure is contaminated for that recipe | **none**: max rank change 1 (α, falcon-and-cc-qc-10p/-20p) and 0.5 (R, c4, dclm-baseline); dR = +25…+62 (≤0.13 SD); Spearman(α, α_clean) = 0.9992, Spearman(R, R_clean) = 0.9998; α shift +0.014…+0.021 (near-constant) | would be exclusion; not triggered |
| C3 | A recipe is an outlier (robust z = (x − median)/MAD_normal **> 3.5**) on any context share (years, entities, code, citations, repdigits, non-round spikes, quantity-like spikes, nonq) | Unusual number *contexts* are a property of the data that could confound a count→β link and should be reported/controlled | **citations:** dclm-baseline-qc-fw-3p (z 6.6), dclm-baseline-qc-7p-fw3 (5.1), dclm-baseline-qc-fw-10p (4.6), fineweb-edu (4.4). (Near-misses, not flagged: dclm-baseline-qc-7p-fw2 3.1, fineweb-pro 2.7 citations; c4 2.8 and falcon-and-cc-qc-10p 2.7 non-round spikes; dclm-baseline 3.4 on c8217.) | data property — **not** a reason to exclude |
| C4 | Probe band whose counts are > 50 % non-quantity excess | Count features evaluated at that band measure a non-quantity use | band 1939–2369 in **all 25** (96.6–98.1 % years); band 9000–10000 0 recipes (8–16 % entities) | universal construct caveat, not per-model |

Exploratory, not used as a rule: the sexagesimal break (`sexagesimal_drop`) gives dclm-baseline-qc-10p z = −4.4,
but the statistic depends on a slope fitted on 30–49 and has no clean interpretation when negative; reported
only for completeness (table column in ctx_full.csv).

Robust z values: `sig2.py` output (second table), reproduced for the flagged column: citations z —
fineweb-pro 2.7, fineweb-edu 4.4, qc-7p-fw2 3.1, qc-7p-fw3 5.1, qc-fw-3p 6.6, qc-fw-10p 4.6, all others ≤ 1.1.

## 4. Measurement validity vs data property — explicit verdict

* **No recipe should be excluded on number-context grounds.** In every recipe at least ~73 % of counts in
  10..9999 are not attributable to any identified non-quantity use, the largest identifiable non-quantity use
  (years) is present in all recipes at 14–26 %, and cleaning it out changes neither α_OLS nor R rankings
  materially (C2). The counts remain a valid proxy for *relative* exposure to numbers across recipes, with the
  caveat that the absolute number is ~15–27 % inflated by years/codes.
* **C3 flags (citation-heavy FineWeb-Edu and the three/four FineWeb-classifier DCLM filters) are genuine
  properties of the data** — what the quality classifier selects (academic, referenced text) — not measurement
  errors. They should be reported and, if desired, controlled (e.g. a family/filter covariate or a sensitivity
  analysis on cleaned counts), never used to drop models.
* **Year share is the context variable that varies most between recipes** (14.5–26.2 %; lowest in the most
  selective falcon and dclm quality filters). This is a property to control: any analysis using raw d4 mass
  or `F20`/`F21` mixes "years seen" with "four-digit quantities seen". The analyst can report R/α on
  year-cleaned counts as a robustness check (the cleaned values are in ctx_full.csv: `R_clean`, `alpha_ols_clean`).
* **C4 is a universal construct caveat:** the fine-β probe band around 2154 sits on the year peak in every
  recipe, so per-band count features (F25/F26-style) at that band measure exposure to *years*. Same for all
  models, so it does not justify exclusion, but band-wise count features should either exclude 1900–2030 or
  say so.
* **Honest limitation:** without raw text (HF Hub blocked, 403) the 0–99 mass — where math, list indices,
  dates, page numbers and decimal fragments live — cannot be decomposed; the `nonq` figures are lower bounds
  for non-quantity use. The between-recipe *differences* in small-number contexts (e.g. F23 0.42–0.60) are
  therefore reported as properties, not as validity failures.
