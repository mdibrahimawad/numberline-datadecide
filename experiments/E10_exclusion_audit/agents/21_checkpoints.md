# 21 — Checkpoint maturity / late-training stability of the number line

Auditor aspect: is the final-checkpoint (step 69369) number line mature and stable for each of the 25
DataDecide 1B models? Blind audit: no alpha–beta relationship was computed or consulted
(`alpha_beta_correlations.json` and `confounders.json` were not opened).

## 0. Bottom line
- **There are no intermediate-checkpoint β measurements for any DataDecide model in the repo.**
  The only non-final folder, `results/beta_fine/step35000-unsharded/`, holds the 6 **Paloma** baselines
  (`allenai_paloma-1b-baseline-*`), and step 35000 is *their final* checkpoint (E09), not a mid-training
  DataDecide checkpoint. E08 (the DataDecide checkpoint test) is "planned, not run".
- So the planned comparison (step ~35000 vs step 69369: β and selected layer) **cannot be made for any of the 25
  models**. Late-training instability is **not assessable**, so no model is flagged.
- What *can* be checked does not show a problem. All 25 models were measured at the same final revision
  `step69369-seed-default` in both E01 and E06. All share one training recipe apart from data: the same step count,
  the same token count and the same LR schedule, a cosine decay to 1 % of peak LR at step 69369. So the models are
  compared at the same annealed end point, which is the right design.
- **Nothing should be excluded on this aspect.** Even if a later measurement showed late drift, that would describe
  how the model and its data learn. It would not be a measurement error, so it should be reported or controlled, not used to exclude.

## 1. Evidence: what checkpoint data exists

| Claim | Evidence |
|---|---|
| Only two β_fine revision folders exist | `ls results/beta_fine` → `step35000-unsharded`, `step69369-seed-default` (plus csv/json files) |
| `step35000-unsharded` is Paloma only | `results/beta_fine/step35000-unsharded/summary.csv`: 6 rows, recipes `allenai_paloma-1b-baseline-{c4,dolma,falcon-refinedweb,mc4,pile,redpajama}`; no DataDecide recipe |
| step 35000 is Paloma's *final* checkpoint | `experiments/E09_paloma_out_of_sample/README.md`: "Final checkpoint revision: **`step35000-unsharded`** for all 6"; `configs/paloma/v1_5-mix-small-mitch-ish-c4-lumi.yaml`: `max_duration: 35_000`, scheduler `cosine_with_warmup, t_warmup 5000, t_max 70000, alpha_f 0.1` (so Paloma stops about halfway through its own cosine, at ≈ 0.61 of peak LR; E09 notes "Paloma's unfinished learning-rate schedule"). This is a *different* model family, so it says nothing about DataDecide maturity. |
| E08 (DataDecide checkpoints) not run | `experiments/E08_checkpoint_test/README.md` line 1–5: "**planned, not run.**"; the folder holds only README.md |
| Intermediate DataDecide checkpoints exist on the Hub but were never used | `docs/infrastructure.md:36-38`: "Intermediate checkpoints exist as other revisions (every 2,500 steps; seeds `default`, plus aux seeds) and are not used yet." |
| No other step/aux-seed result anywhere | `grep -rlE "step[0-9]{4,5}-seed|seed-(large|small|aux)|step(17500|35000|52500|67500|7500)" results src runpod_jobs` (Paloma excluded) → only files that contain the string `step69369-seed-default`; `git branch -a` → only `main` and `claude/relaxed-keller-a4rdiz`; no checkpoint results in git history |
| Training-seed noise also unmeasured | `docs/handover.md:58`: "Training-seed noise: β of the aux-seed final checkpoints … " listed as to-do |
| Hub could not be queried from this audit | `curl https://huggingface.co/api/models/allenai/DataDecide-c4-1B/refs` → proxy `connect_rejected` (403). New measurements were impossible here anyway (no GPU). |

## 2. Evidence: same final step, same schedule, correct revision

| Claim | Evidence |
|---|---|
| All 25 E01 results come from the final revision | python over `results/datadecide/<recipe>.json` → field `revision == "step69369-seed-default"` for all 25 (the 2 other json files in the folder are the excluded summary files) |
| All 25 E06 results come from the final revision | python over `results/beta_fine/step69369-seed-default/<recipe>.json` → `revision == "step69369-seed-default"` for 25/25 |
| Same final step and tokens for every recipe | `docs/datadecide_sampling.md:26-30`: global batch 704 × 2048, `max_duration = 100000000000T`, final checkpoint `step69369` = 100.02B tokens; `configs/datadecide_models.json`: one `revision` for all recipes |
| Same LR schedule for every recipe | OLMo `DataDecide` branch `scripts/ladder.py` (fetched from raw.githubusercontent.com into `/tmp/claude-0/-home-user-numberline-datadecide/47026aba-bee6-5ac1-8297-97a27f9ca992/scratchpad/dl21/ladder.py`), line 207: `lr = 0.0047 * (model_size / 108000000) ** (-1 / 3)` (depends only on model size); lines 241-246: `SchedulerConfig(name=cosine_with_warmup, alpha_f=0.01, warmup_min_lr=0.0, t_warmup=round(model_size/(global_batch_size*max_seq_len)))`; `max_duration=f"{length_in_tokens}T"`. Nothing in it depends on `--data`. So all 25 recipes share the peak LR, the warmup (≈ 820 steps) and the cosine to 1 % of peak at the 100B-token end. Caveats: (a) `t_max` is not set, and I take it to default to the run length, as in OLMo's trainer; I did not check that in this session. (b) The docs say the released runs may come from a sibling branch (`benb/ladder-1xC`, `docs/datadecide_sampling.md:36`), which `docs/datadecide_sampling.md` treats as equivalent for these settings. |
| LR at late steps (cosine, warmup 818, alpha_f 0.01; my computation) | fraction of peak LR: step 17500 → 0.862, 35000 → 0.507, 52500 → 0.151, 62500 → 0.034, 67500 → 0.012, 69369 → 0.010 |
| Difference that does exist: data repetition | `experiments/E04_exact_100b_alpha/DETAILS.md` ("share of recipe read"): fineweb-pro 121 %, dclm-baseline-qc-10p 122 %, dclm-baseline-qc-fw-10p 102 %. These 3 recipes go into a partial second epoch (reshuffled with seed + 1); all others read < 100 % once. The step count and schedule are the same; only the amount of repeated data differs. This is a property of the data recipe. |

Implication: at step 69369 every model has finished an identical cosine anneal. The LR is 1 % of peak, and it was under 3.5 % of peak over the last ~7,000 steps, so each update moves the weights little at the end. The models are compared at a matched, fully annealed point. That supports the β values being end-of-training values, but it does not prove the β values have stopped changing: that needs checkpoint measurements (§3).

## 3. Exclusion criteria (fixed before looking at any checkpoint data, none of which exists)

Measurement: β_cont at a **fixed layer** (the layer E06 selected at step 69369), E06 protocol (3 prompt seeds), at
checkpoint step 67500 (the last released revision before 69369, every 2,500 steps) and at about 50 % (step 35000).

- **C1 — late instability (exclusion candidate):** |β_cont(69369) − β_cont(67500)| > max(0.05, 3·√(sd₁² + sd₂²)).
  *Why this threshold:* the E06 seed s.d. is ≈ 0.015 (median; `experiments/E06_fine_beta/README.md`), so 3σ of a
  difference is ≈ 0.06. The floor of 0.05 is ≈ 6 % of the 0.55–1.38 cross-model range. Over those last 1,869 steps the LR
  is only 1.0–1.2 % of peak (§2), so a change this large would mean the end point is not a stable read-out.
- **C2 — late layer instability:** the selected layer at 67500 differs from 69369 by > 1 layer **and** the
  √(EV·|ρ|) score margin at 69369 is ≥ 5 % (not a near tie; E06 shows near-tie switches of 0.7–7.7 % are common and
  belong to the layer-selection audit, not this one).
- **C3 — mid-to-final drift (report only, NOT exclusion):** |β_cont(69369) − β_cont(35000)| > max(0.10, 3σ).
  Between 50 % and 100 % of training the LR is still 51 %→1 % of peak and the model sees 50B more tokens, so a change
  there is expected learning. It shows whether β is "converged", but every model is measured at the same matched
  end point, so this is a property to report or control (e.g. as a covariate), not a reason to drop a model.

**Result of applying C1–C3: no model is flagged under any of them, because no DataDecide model has a non-final
measurement. All 25 are "not assessable".**

## 4. Per-model table (all 25)

Values come from `results/datadecide/alpha_beta.csv` (E01 `selected_layer`, `beta_direct_mean`) and
`results/beta_fine/step69369-seed-default/summary.csv` (E06 `layer`, `beta_cont`, `beta_cont_std`). Rev = revision
in each per-model json. "Mid/late ckpt" = intermediate measurement present? Epoch = share of recipe read (only the 3 > 100 % are marked).

| recipe | E01 layer | E01 β_direct | E06 layer | E06 β_cont (sd) | Rev E01/E06 | final step | Mid/late ckpt | C1 | C2 | C3 | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c4 | 7 | 0.620 | 3 | 0.596 (0.013) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline | 8 | 0.666 | 8 | 0.584 (0.016) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-25p-dolma1.7-75p | 8 | 1.108 | 8 | 1.101 (0.027) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-50p-dolma1.7-50p | 8 | 0.952 | 8 | 0.831 (0.018) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-75p-dolma1.7-25p | 10 | 0.841 | 10 | 0.762 (0.018) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-qc-10p | 8 | 0.717 | 8 | 0.657 (0.017) | 69369/69369 | 69369 | none | n/a | n/a | n/a | 2nd epoch (122 %) |
| dclm-baseline-qc-20p | 9 | 0.760 | 9 | 0.717 (0.011) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-qc-7p-fw2 | 8 | 0.838 | 8 | 0.669 (0.009) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-qc-7p-fw3 | 9 | 0.750 | 3 | 0.806 (0.017) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dclm-baseline-qc-fw-10p | 3 | 0.809 | 3 | 0.756 (0.008) | 69369/69369 | 69369 | none | n/a | n/a | n/a | 2nd epoch (102 %) |
| dclm-baseline-qc-fw-3p | 9 | 0.784 | 2 | 1.065 (0.053) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dolma1_6plus | 9 | 1.077 | 9 | 1.008 (0.019) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dolma1_7 | 6 | 0.978 | 4 | 0.831 (0.011) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dolma1_7-no-code | 8 | 0.856 | 8 | 0.773 (0.017) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dolma1_7-no-flan | 9 | 1.197 | 9 | 1.191 (0.022) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dolma1_7-no-math-code | 4 | 0.618 | 4 | 0.552 (0.013) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| dolma1_7-no-reddit | 8 | 1.360 | 8 | 1.176 (0.018) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| falcon | 9 | 0.792 | 8 | 0.672 (0.010) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| falcon-and-cc | 9 | 0.649 | 9 | 0.626 (0.027) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| falcon-and-cc-qc-10p | 8 | 0.713 | 3 | 0.876 (0.006) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| falcon-and-cc-qc-20p | 8 | 0.739 | 8 | 0.678 (0.020) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| falcon-and-cc-qc-orig-10p | 8 | 0.771 | 3 | 0.701 (0.009) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| falcon-and-cc-qc-tulu-10p | 9 | 0.948 | 2 | 1.380 (0.002) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| fineweb-edu | 9 | 0.895 | 9 | 0.872 (0.017) | 69369/69369 | 69369 | none | n/a | n/a | n/a | |
| fineweb-pro | 7 | 0.662 | 7 | 0.577 (0.020) | 69369/69369 | 69369 | none | n/a | n/a | n/a | 2nd epoch (121 %) |

(69369 = `step69369-seed-default`.) The E01 ≠ E06 layer differences (8 models) are within one checkpoint, so they
are a layer-selection near-tie issue (E06 README) and not evidence about training maturity. I leave them to the
layer-selection audit.

## 5. Is any of this a validity problem?

| Item | Validity problem? | Verdict |
|---|---|---|
| Missing intermediate checkpoints (all 25) | No; it is a **gap in evidence**, not a defect in any model's measurement. It means the claim "β is converged at the end of training" is untested, and the paper should not make that claim without E08-style data. | Report as a limitation; do not exclude |
| Hypothetical late drift (C1/C2, if measured later) | Only C1/C2 (change in the last 2,500 steps at ≈ 1 % LR) would make the step-69369 value an unreliable read-out, which is a measurement-validity issue. With no data, no model meets it. | No exclusions |
| Mid-to-final change (C3) | No. All models are compared at the same annealed end point, and how far β moves during training is a property of model and data. Excluding "slow converging" models would select on the outcome. | Report / control, never exclude |
| Partial second epoch (fineweb-pro, dclm-baseline-qc-10p, dclm-baseline-qc-fw-10p) | No. Same step, tokens and schedule; repeating data is part of the recipe, and the counts already account for it (E04 reproduces the second epoch). | Report as a data property |
| Paloma step35000 being mid-schedule (≈ 0.61 of peak LR) | Not relevant to the 25 DataDecide models. It is a caveat for E09 (Paloma β comes from an un-annealed checkpoint and may not be comparable with the annealed DataDecide models). | Note for E09 only |

**Honest note:** on checkpoint maturity, nothing should be excluded. The repo cannot show that any model's number line is
still changing at step 69369, because no DataDecide intermediate checkpoint was ever measured. The training setup
(the same 69,369 steps and the same cosine decay to 1 % of peak LR for every recipe) makes the final checkpoints a well-matched and
annealed comparison point. The cheapest way to close the gap: E06 protocol on revision
step67500 (and step35000) of all 25 models, then apply C1–C3 as written above.
