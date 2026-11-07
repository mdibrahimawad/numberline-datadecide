# E13 METHODOLOGY: how LLMs represent integers (extraction on RunPod + full test panel)

This file is a complete work order for my coding Claude. Read all of it first. Work stage by stage. After each stage,
stop, send me the results, and wait. Everything marked PRE-REGISTER must be written into a PREREG file and committed
BEFORE the real data is analysed.

Repo: mdibrahimawad/numberline-datadecide. New work goes in `experiments/E13_number_code/` (README.md, PREREG_*.md,
src/, results/). Reuse `src/e12_manifold.py`, `src/e12b_smooth_digits.py`, `src/e12c_heldout_r2.py`.

---------------------------------------------------------------------------------------------------
## PART 0. WHY THIS EXPERIMENT (short)

Old vectors: last-token state at the final "=" of `A=A,B=B,C=C,D=D,n=` (the model is about to COPY n). The papers we follow
(Levy & Geva; Stefanik/Kadlcik; Kantamneni & Tegmark; Zhu et al.; Gurnee & Tegmark) read the vector AT the number's own last
token, in natural or varied contexts, average over contexts, keep tokenization fixed, validate with probes, and (the best ones)
with causal edits. Our old results (E12/E12b/E12c) found: dimension and homology tests have no power; a smooth function of n
predicts random held-out numbers but not a held-out leading-digit block; digits add up like categories; Stage 1 of the
hypothesis plan failed. Two surprises may be caused by the copy prompt: (1) in digit-splitting models the first digit loses its
order in late layers (it may just be "which digit do I type next"); (2) 199 and 200 are closer than expected.
This experiment re-reads the same numbers at better positions and in more contexts, then runs the full test panel.

---------------------------------------------------------------------------------------------------
## PART 1. GLOBAL RULES

G1. Sets by digit length, never mixed: S2 = 10-99, S3 = 100-999, S4 = 1000-9999 (S1 = 1-9 only as a sanity set).
    For chunk tokenizers (Pythia, OLMo-2, Llama-3) keep only numbers that are exactly ONE token in the context (with its leading
    space). Record n_tokens for every number; drop numbers whose token count differs from the set's modal count.
G2. Models. Stage A: Pythia-1.4B (EleutherAI/pythia-1.4b) and Qwen2.5-1.5B (Qwen/Qwen2.5-1.5B). After the gate: Crystal,
    OpenLLaMA-3B, StarCoderBase-3B, SmolLM2-1.7B, OLMo-2-1B (allenai/OLMo-2-0425-1B), RedPajama-3B, Llama-3.2-1B (gated: needs an HF token).
G3. Discover on Qwen + Pythia, lock thresholds, then test on the rest. A result is "supported" only if it holds in >= 2 tokenizer
    families (digit-splitting AND chunk).
G4. Controls in every test: shuffled n; matched-smoothness fake; digit-label shuffle (permute digit values inside a position);
    PLANTED-STRUCTURE POWER CHECK (plant the claimed structure in synthetic data at the real noise level; if recovered < 80% of
    the time, report "no power" and make no claim from that test).
G5. Dev/test split of numbers: a fixed 20% random "dev" set (seed 13) is the ONLY place to choose layers, bandwidths, ranks and
    subspace sizes. Report all layers anyway. Never pick the best-looking layer on test data.
G6. Uncertainty: bootstrap over numbers (and over contexts). Report effect sizes with 95% CIs.
G7. Report everything: all models, all stored layers, failures, tests with no power. Label results PRIMARY (pre-registered) or EXPLORATORY.
G8. Prep: centre, z-score each dimension, PCA to 50 fitted on TRAINING numbers only. Robustness rows: no z-scoring; top-3 raw
    variance dimensions removed.
G9. Seeds fixed and recorded. Do not change seeds between a main run and a rerun.

---------------------------------------------------------------------------------------------------
## PART 2. EXTRACTION PROTOCOL

### 2.1 Contexts (templates). {n} is the integer written in plain digits, no commas. Always a single space before {n} unless noted.
 C1 copy      : `A=A,B=B,C=C,D=D,{n}=`   (demonstrations: 4 random letters? keep EXACTLY the E10 format, seed 42; plus a second set with seed 7)
 C2 apples    : `She bought {n} apples at the market.`
 C3 population: `The population of the town is {n}.`
 C4 room      : `Room {n} is on the third floor.`
 C5 compare   : `Which is larger, {n} or 500? Answer:`
 C6 successor : `{n} + 1 =`            (number at the start: no leading space; note this in the metadata)
 C7 list      : `3, 8, {n}, 21, 40`
 C8 price     : `The price is {n} dollars.`
Order the contexts in the file exactly as above. Keep the number of tokens BEFORE {n} identical across the numbers inside one context.

### 2.2 Read positions (for every context)
 P_num   : hidden state at the LAST token of {n}. In C1 this is the token right before the final "="; it is the same prompt as the old
           data, only read one step earlier. This is the single most informative comparison (copy plan vs the number itself).
 P_next  : the token right after {n} (space, period, "or", "+", "=", ",", " apples" etc.). In C1 this is the final "=" (the OLD reading).
 P_task  : only for C5 (the final ":" of "Answer:") and C6 (the final "="), where the model is about to answer. Used for behaviour links.
Find positions with the fast tokenizer's `return_offsets_mapping=True`: locate the char span of {n} inside the prompt; P_num = the last
token whose span ends inside that span; P_next = P_num + 1. Assert the decoded tokens of P_num cover the last character of {n}.

### 2.3 Layers to store
hidden_states from HF (`output_hidden_states=True`) has length L+1: index 0 = embedding output, index L = last layer (for GPT-NeoX/Pythia
and many models the last element has the FINAL NORM applied; note this in the metadata). Store these indices:
  {0, round(0.125L), round(0.25L), round(0.375L), round(0.5L), round(0.625L), round(0.75L), round(0.84L), L}
(9 layers). Additionally store, for P_num only, the context-AVERAGED vector at ALL layers (fp16).

### 2.4 What to save per model (directory `/workspace/e13_out/<model>/`)
 - `meta.csv`: n, set (S1..S4), n_tokens per context, token ids of the number, P_num and P_next index per context, ok flag, copy-correct flag for C1.
 - `vec_<ctx>_<pos>.npy`: float16, shape [N, 9, d] for ctx in C1..C8, pos in {num, next} (C5/C6 also `task`).
 - `vec_avg_num_alllayers.npy`: float16, shape [N, L+1, d] = mean over C2..C8 of the P_num vectors (exclude C1 so it stays independent).
 - `static_embed.npy`, `static_unembed.npy`: input embedding rows E(token) and unembedding rows U(token) for the single-token numbers,
    both for the " 347" (leading space) and "347" (no space) forms when they are single tokens; with their token ids.
 - `config.json`: model id, revision/commit hash, dtype, torch/transformers versions, tokenizer name, templates, seeds, layer indices.
Size estimate (d=2048, N=10,000, fp16): 8 ctx x 2 pos x 9 layers = about 5.9 GB per model, plus ~2 GB for the all-layer average. Keep files per context so a failure does not lose everything.

### 2.5 Correctness checks (do before the full run; stop if any fails)
 a. Batch vs single: for 50 random prompts, max relative difference between batched and one-by-one hidden states <= 2e-3 (fp16/bf16) at every stored layer.
 b. Position check: print 20 random prompts with the tokens at P_num and P_next.
 c. Copy check: fraction of C1 prompts where the model's greedy next token equals the first token of n (report; E10 had 100% on the final "=" being the last token).
 d. Reproduce an old number: for the C1/P_next vector at 5 random n, cosine similarity with the stored old vector >= 0.999 at the matching layer (same model, same prompt text, same seed 42). If not, find out why before going on.
 e. Padding: use right padding with an attention mask and gather by position index, or left padding with adjusted indices. Test (a) covers it.

### 2.6 Code sketch (adapt, do not trust blindly)
```python
import torch, numpy as np, json
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).cuda().eval()   # bf16 for Qwen, fp16 ok for Pythia
def find_positions(prompt, n_str):
    enc = tok(prompt, return_offsets_mapping=True)
    s = prompt.index(n_str); e = s + len(n_str)            # the number must occur once, check this
    idx = [i for i,(a,b) in enumerate(enc["offset_mapping"]) if a < e and b > s]
    return enc["input_ids"], idx[0], idx[-1], idx[-1] + 1     # first/last token of the number, next token
@torch.no_grad()
def run_batch(prompts, layers):
    ... tokenise with padding, forward with output_hidden_states=True, gather [b, layer, pos] -> float16
```
Sort prompts by length to batch efficiently. Batch size 128-512 on a 24-48 GB GPU. Use `torch.inference_mode()`.

### 2.7 RunPod setup
 1. Pod: 1x RTX 4090 (24 GB) or A40 (48 GB) is enough for models up to 3B in bf16/fp16. Template: PyTorch 2.x CUDA. Volume: 100 GB at /workspace.
 2. `pip install transformers accelerate scipy scikit-learn pandas matplotlib ripser` (torch is preinstalled). Set `HF_HOME=/workspace/hf`.
 3. Clone the repo into /workspace. Run inside `tmux` so a disconnect does not kill the job.
 4. Order: unit checks (2.5) -> Pythia-1.4B -> Qwen2.5-1.5B. Estimated time: 80,000 short prompts per model, roughly 10-30 minutes each on these GPUs (an estimate; measure it) -> about $1 per model.
 5. After each model: upload `/workspace/e13_out/<model>/` to a PRIVATE Hugging Face dataset repo (or the Modal volume used before) and verify the file sizes. Then STOP the pod. Never paste tokens or keys into the chat or the repo.

---------------------------------------------------------------------------------------------------
## PART 3. THE TEST PANEL (run on every context x position x model x layer x set; use the old data for comparison)

### Common metrics
 M1 Smooth part: kernel smoother f(n), bandwidth by 5-fold CV from {1,2,5,10,20}% of the range, MINIMUM 1% of the range. Held-out R2:
    random 80/20 (3 repeats) and held-out block (70-79 / 700-799 / 7000-7999). R2 = 1 - sum||Z_test - pred||^2 / sum||Z_test - mean(Z_train)||^2, with standardisation and PCA fitted on training numbers only.
 M2 Additive digit part: one-hots for units, tens, hundreds, thousands (centred); backfitting with f; report variance shares of f, digits, residual; and held-out R2 of "curve + digits".
 M3 Noise ceiling: split the contexts C2..C8 into two random halves, average each half, and compute the same R2 between halves. Report every R2 as a FRACTION of this ceiling.
 M4 Probes (held-out numbers): (i) ridge linear probe for n and for log n (R2, Spearman); (ii) sinusoidal probe (sum of sines with learned frequencies, as Kadlcik/Stefanik); (iii) circular probe per digit position (2-D projection to (cos, sin) of 2*pi*d/10, decoded by atan2; accuracy); (iv) linear classifier per digit position (accuracy).

### H1 - position-resolved digit code ("floating-point code")
 For each digit position p (units, tens, ...): the 10 centred digit-effect vectors c_v from M2 (other positions regressed out). Compute
   O1 = Spearman(pairwise centroid distances, |a-b|) over the 45 pairs;
   PR = participation ratio of the 10 centroids (1 = line, ~2 = circle, ~9 = simplex);
   TEMPLATE FIT = cross-validated fit of three templates to the 45 distances: line (|a-b|), circle (chord of 2*pi*|a-b|/10), simplex (constant), free scale;
   AXIS = share of centroid variance on the first axis x |Spearman(axis projection, d)|.
 Also: is the NUMBER OF DIGITS (exponent) linearly decodable (held-out-length probe: train on lengths {2,3}, test on length 4 or 1)?
 Controls: digit-label shuffle; planted line / circle / simplex centroids at the real noise.
 PRE-REGISTER: support = (O1_leading - O1_units) > 0.30 with 95% CI above 0; O1 non-increasing from leading to units in >= 70% of model x depth cells at depth >= 50%; in >= 2 tokenizer families.
   Rival R-a (uniform ordinality): all O1 > 0.5 and range < 0.1. Rival R-b (pure symbolic): all O1 < 0.3. Rival R-c (units digit is a circle): units template = circle in >= 70% of cells.

### H2 - carries, successor operator, tree vs line
 H2a Step length s(n) = ||z(n+1) - z(n)|| (z = standardised PCA-50). Fit s(n) = smooth trend + additive-model prediction + sum_j a_j * 1[carry level >= j] (carry level = number of trailing 9s). Test a_1 > 0 and a_2 > a_1.
 H2b Successor operator: ridge T on steps of carry level 0 (training numbers only); held-out R2 for z(n+1) from z(n) at carry level 0, 1, 2, 3.
 H2c Tree vs line: for all pairs fit E||z_m - z_n||^2 with (a) isotonic g(|m-n|) (line), (b) sum over prefix levels w_l * 1[prefixes differ at level l] (tree), (c) both. Cross-validate on held-out numbers AND on held-out whole leading-digit groups. Report the ultrametric-violation index, the Gromov delta (divided by diameter) of prefix-group centroids, and the isotonic R2 INSIDE each leading-digit group.
 Controls: shuffle carry labels; planted smooth curve with and without carry jumps at real noise; matched-smoothness fake.
 PRE-REGISTER: H2a supported if a_1 > 0 and a_2 > a_1 with CIs excluding 0 in >= 70% of cells and >= 2 families. H2b: R2(level 0) - R2(level >= 1) > 0.20. H2c: tree beats line on held-out GROUPS by >= 0.10 R2 in late layers of digit-splitting models (report the opposite if Pythia mid layers prefer the line).

### H3 - quantity / composition / lexicon
 Residual r(n) = z(n) - f(n) - D(n) (M1, M2). Reliability = correlation of r(n) between the two context halves (per number, then averaged), after removing the shared mean.
 Regress ||r(n)|| and vector length on log SlimPajama frequency (partial out digit length) and on flags: year 1900-2030; multiple of 100 and of 1000; repdigit; power of 2; famous (007, 404, 911, 666, 1337).
 Re-run the E10/E11 rare-vs-frequent crowding analysis with r(n) removed.
 PRE-REGISTER: lexicon exists if reliability >= 0.30 and the frequency slope CI excludes 0 in >= 2 families; crowding is "lexical" if its effect size falls by >= 50% after removing r(n).

### H0 - invariance (what the new data is for)
 For every metric above compare: P_num vs P_next within each context; C1 P_num vs C1 P_next (copy plan vs number); the 7 natural contexts vs each other; the context-average; the static embedding E(n) and unembedding U(n).
 Report a matrix (metric x condition) as a fraction of the ceiling.
 PRE-REGISTER: invariance holds if each key metric on C1/P_num and on >= 3 of the 7 natural contexts is >= 50% of its value at C1/P_next and has the same sign and ordering across models. If not, the conclusion is "the old structure was mostly the copy plan / output geometry"; report which parts survive.

### P7 - FIXED POSITIVE CONTROL (must pass before any topology/shape claim)
 Days of the week (7) and months (12): 20 prompt templates each (e.g. `Today is {day}.`, `The meeting is on {day}.`, `Her birthday is in {month}.`), read at the last token of the item, averaged over templates. Project onto the top-k PCs (k = 2..6) first. Score: circular-order score (do the two nearest items in the projection equal the calendar neighbours?) >= 0.75, plus a circular probe. The detector must find the days/months circle in >= 2 of 5 depths in each model. If it does not, drop persistent-homology and loop claims from the paper and state why.
 Run the SAME detector on the number sets (units digit mod 10) and report it.

### P8 - global dimension knee (replaces local ID)
 Smooth k-dimensional latent model for k = 1..10, fit on 80% of numbers, held-out R2 (random + block). Controls: shuffled n; fake flat sheet. PRE-REGISTER: knee = smallest k reaching 90% of the k=10 R2; it must be sharper than in the controls. Expected under H1 (about 1 for magnitude + ~9 per categorical digit position) - report the expected value too.

---------------------------------------------------------------------------------------------------
## PART 4. CAUSAL STAGE (only if Stage A shows H1 or H2 holds on the number-token reading)

Models: Pythia-1.4B, Qwen2.5-1.5B, OLMo-2-1B.
 Subspaces (training split, layer chosen on the dev split): S_mag = top-m directions of f(n) (m by CV, about 3-5); S_dig(p) = span of the 9 centroid differences of position p; S_carry = span of the carry effects of H2. Controls: random subspaces of equal dimension, orthogonal to the others; subspaces fitted on shuffled n.
 Interventions (forward hooks at layer l, at the number-token positions): (i) project-out ablation; (ii) activation patching of only that subspace between a clean and a corrupted number; (iii) steering: add alpha * (f(n') - f(n)).
 Behaviours (500+ items each, balanced for carries and digit length; metric = logit difference): B1 which of two numbers is larger; B2 the last digit of n; B3 the leading digit of n; B4 n + 1; B5 a + b with and without carries; B6 copy n.
 Prediction: S_mag -> B1 (B3 partly); S_dig(units) -> B2, B6; S_carry -> carry cases of B4/B5 only.
 PRE-REGISTER: target effect >= 2x the random-subspace effect AND off-target effect <= 1.2x random, with bootstrap CIs over items, in >= 2 models. Report the full subspace x behaviour effect matrix including failures.

---------------------------------------------------------------------------------------------------
## PART 5. TRAINING AND DATA STAGE (last)

 5a Checkpoints: Pythia-1.4B (16 checkpoints) and OLMo-2-1B (12): extract with the SAME protocol (contexts C1..C8, P_num), run the whole panel, add the behaviour accuracies (copy, comparison, successor, addition with and without carries). Emergence time = the step where a metric reaches 50% of its final value, with bootstrap CIs. Predicted order: unit-digit categories -> leading-digit ordinality -> carry structure -> lexicon. PRE-REGISTER the order; test with a permutation test over metrics (p < 0.05 for >= 3 metrics in both models).
 5b DataDecide: all 25 recipes (one tokenizer, same architecture, 100B tokens). Regress each panel metric on the recipe's numeral frequency alpha (recipe-level bootstrap). Prediction: spacing/ordinality decay rate and lexical size vary with alpha; template type and carry spikes do not.
 5c Optional (needs compute): 4-6 small models (~160M) pretrained with numeral frequency deliberately changed.

---------------------------------------------------------------------------------------------------
## PART 6. ORDER OF WORK AND GATES

 Stage A1 (RunPod): checks 2.5, extract Pythia-1.4B + Qwen2.5-1.5B, store, upload.            -> send me a short status.
 Stage A2 (analysis): P7 positive control, then M1-M4, H0, H1, H2, H3, P8 on those two models.  -> send results; STOP.
 GATE A: (i) positive control passes? (ii) does H1/H2 hold on the number-token reading? If neither, switch to the negative-result paper.
 Stage B: the other 7 models (same protocol), replicate.                                              -> send results; STOP.
 Stage C: causal (Part 4).                                                                            -> send results; STOP.
 Stage D: training and data (Part 5).                                                                 -> send results.
 Stage E: freeze, one verdict table (supported / not supported / cannot tell) with numbers, figures, and a list of deviations.

---------------------------------------------------------------------------------------------------
## PART 7. WHAT TO SEND BACK AFTER EACH STAGE

 - One page: question, what was run, verdict for each hypothesis with numbers and the power check.
 - Tables (CSV) for all models, layers, sets, contexts, positions and controls; the commit hashes of PREREG and code; a list of deviations.
 - Figures: ordinality by digit position (per model x depth); step length vs n with carry spikes; successor-operator R2 by carry level; tree vs line bars on held-out groups; the invariance matrix; the positive-control result; R2 vs k.
 - Anything surprising and anything that contradicts my earlier results.

---------------------------------------------------------------------------------------------------
## PART 8. RISKS (tell me if you see them)

 - Estimators with no power (local dimension, homology): not used for claims unless the positive control passes.
 - A strong result at layer 0 may just be the embedding table; always compare with E(n).
 - Chunk-tokenizer sets are small after keeping single-token numbers: widen CIs and say so.
 - Context-average removes real context effects; keep per-context results too.
 - The additive digit model and f(n) can compete for the same variance in 2-digit sets: also run the units-only variant.
 - Over-tuning: layers, bandwidths and subspace sizes only on the dev split.
 - Novelty must be checked by literature search before any claim (per-digit-position geometry, carry/successor geometry, tree structure, lexical component, double dissociation). Closest: Kantamneni & Tegmark (arXiv 2502.00873), Levy & Geva (2410.11781), Bassi & Tomar (2607.04167), Stefanik et al. (2510.26285), Kadlcik et al. (2506.08966), Karkada et al. (2602.15029).
