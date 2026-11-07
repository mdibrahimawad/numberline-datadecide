# PROMPT FOR MY OTHER CLAUDE: an A*-level study of how LLMs represent integers

You are my research partner. You have my repo (mdibrahimawad/numberline-datadecide), my hidden-state vectors
(`topology_number_line`, `data/`, `data/e11_ckpt/`), and code for models up to ~1.5B. Work in stages. After each stage,
STOP, send me the results, and wait. Do not skip ahead.

---------------------------------------------------------------------------------------------------
## 0. WHERE WE ARE (what is already known, so you do not redo it)

Question from my professor: "prove there is a 1-D manifold that encodes the number line in an LLM."
That claim is empty as stated (any finite set lies on some curve), so I tested stricter versions (E12, E12b, E12c).

Setup: vectors are the last-token hidden state at the final "=" of `A=A,B=B,C=C,D=D,n=` for n = 1..10,000, 5 depths
(~25/50/75/84%/last), 9 models (Crystal, OpenLLaMA-3B, StarCoderBase-3B, Qwen2.5-1.5B, SmolLM2-1.7B, Pythia-1.4B,
OLMo-2-1B, RedPajama-3B, Llama-3.2-1B). Checkpoints: Pythia (16), OLMo-2 (12), 2,100 numbers. DataDecide: 25 recipes.

Findings so far (all with pre-registration and controls):
1. Local-dimension and persistent-homology tests have NO POWER at the real noise level (fake curves are not
   distinguished from fake sheets; the days/months positive control also failed). So "no holes" and "ID = 17-33" cannot
   support or refute a 1-D claim. Do not use those tests as evidence again.
2. A smooth function of n predicts randomly held-out numbers (held-out R2 up to 0.78 in Qwen late layers, ~0 for
   shuffled n). But it FAILS on a held-out leading-digit block (700-799): R2 < 0 in 25 of 30 cells.
3. An additive digit model (units + tens + ...) explains distances better than numeric closeness in 30/30 datasets.
   Digit values behave more like near-equidistant categories (simplex) than like a line (E11 notes).
4. Real data is not "just smoothness" (differs from the matched-smoothness fake). Periodic structure at periods 2/5/10 is small.
5. Spacing is near-linear (power p ~ 0.75-1.2), not log, once digit length is held fixed.
6. The number line forms abruptly in training (Pythia between steps 2,000 and 4,000), together with the copy skill.
   Digit-splitting tokenizers give 9 leading-digit regions late; chunk tokenizers give a smooth line.

Working picture: "a coarse number line + an additive digit code + a large leftover". I want to turn this into a
sharp, causal, A*-level paper. The stages below test hypotheses that, as far as I know, nobody has tested before.
(Novelty must be CHECKED by literature search before I claim it. See section 9.)

---------------------------------------------------------------------------------------------------
## 1. A CONFOUND WE MUST FACE FIRST (this changes how every result is read)

Our vectors are taken at the "=" token AFTER n, in a COPY prompt. The model's next job is to output n again.
So the state mixes (a) what the model knows about n and (b) what it is about to write.
- In digit-splitting models the next token is the leading digit of n. The "9 leading-digit regions" in late layers may
  simply be next-token (output) geometry.
- In chunk tokenizers the next token is the number token itself, so late-layer geometry may mostly be the
  unembedding geometry of number tokens.
- The "number line appears together with the copy skill" result fits this reading too.
If the structure is mostly output geometry, the headline "LLMs have a number line" shrinks to "LLMs have number-shaped
output logits". That would still be a finding, but a different one. So Stage 2 (invariance check) is mandatory.

---------------------------------------------------------------------------------------------------
## 2. RULES FOR EVERY STAGE (non-negotiable)

R1. PRE-REGISTER. Before running anything, write the stage's question, metrics, thresholds and decision rules in
    `experiments/E13_*/PREREG_<stage>.md`, commit, then run. List every deviation afterwards.
R2. SETS. Never mix digit lengths. Use 10-99, 100-999, 1000-9999 as separate sets. For chunk tokenizers use only
    numbers that are a single token within the set. Report the token count of every set.
R3. REPLICATE. Discover on Qwen2.5-1.5B and Pythia-1.4B. Lock thresholds. Test on the other 7 models.
    A result counts as "supported" only if it holds in at least 2 tokenizer families (digit-splitting AND chunk).
R4. CONTROLS in every test: shuffled n; matched-smoothness fake; digit-label shuffle (permute digit labels inside a
    position); and a PLANTED-STRUCTURE POWER CHECK (plant the claimed structure into synthetic data at the real noise
    level. If the test recovers it < 80% of the time, report "no power" and make no claim).
R5. UNCERTAINTY. Bootstrap over numbers (and over prompts when several exist). Report effect sizes with 95% CIs.
R6. REPORT EVERYTHING. All models, all 5 depths, failures, tests without power. No picking the best layer.
R7. LABEL results PRIMARY (pre-registered) or EXPLORATORY. Exploratory results cannot be headline claims.
R8. Standardise (centre, z-score), PCA to 50 fitted on training numbers only. Repeat key results without z-scoring and
    with the 3 largest-variance raw dimensions removed.

---------------------------------------------------------------------------------------------------
## 3. THE HYPOTHESES

### H1: "Floating-point code" (ordinality decays with digit position)
Motivation. Prior work asks "is the number line linear, log, or a helix?" (Kantamneni & Tegmark; Levy & Geva; log-compression
papers). Nobody, to my knowledge, asked the position-resolved question: for EACH digit position, is the model's code for
the 10 digit values (a) an ORDERED line, (b) a CIRCLE (period 10), or (c) a SIMPLEX of equidistant categories?
My E11 notes say digits behave mostly like categories (distance sd/mean 0.06-0.19) with a weak order (Spearman up to +0.77
for the tens digit). That hints that different positions might use different codes.
Claim. Integers are stored like scientific notation: the NUMBER OF DIGITS (exponent) is explicit; the LEADING digits
carry magnitude and are ORDINAL; trailing digits are CATEGORICAL (simplex). Ordinality decays from the leading to the units digit.
Rivals. (R-a) all positions equally ordinal ("positional value" code); (R-b) all positions categorical (pure symbolic);
(R-c) units digit is a circle everywhere (helix story); (R-d) ordinality depends only on tokenizer, not on position.
If right: a compact law ("ordinality(p) ~ geometric decay") plus an explanation of why arithmetic errors concentrate in
low digits and why the block holdout fails. If wrong: the clean alternative R-a or R-b is itself a strong, unexpected result.
Method.
 1. Fit the additive digit model per set. For each position p, take the 10 digit-effect vectors c_v (v = 0..9), averaged
    over the other digits (regress the other positions out first).
 2. Per position compute: O1 = Spearman between pairwise centroid distances and |a-b| (45 pairs);
    PR = participation ratio of the 10 centroids (1 = line, ~2 = circle, ~9 = simplex);
    template fit = cross-validated fit of three templates to the 45 distances (line: |a-b|; circle: chord of 2*pi*|a-b|/10;
    simplex: constant), with free scale; and a "best 1-D axis" score = share of centroid variance on the first axis times
    |Spearman(axis projection, v)|.
 3. Do it for all models and depths, sets of 2, 3 and 4 digits, and (later, Stage 6) over training checkpoints.
 4. Also measure the exponent code: is "number of digits" linearly decodable at the "=" state (use 1-9, 10-99, 100-999, 1000-9999 in
    a model where lengths can be compared, with a held-out-length probe).
Controls. Digit-label shuffle (destroys ordinality, keeps variance); planted line/circle/simplex centroid sets at real noise.
Decision rule (adjust BEFORE running, then commit). Support = (O1_leading - O1_units) > 0.30 with 95% CI above 0, O1
monotone non-increasing from leading to units in >= 70% of model x depth cells at depth >= 50%, in >= 2 tokenizer families.
R-a if all O1 > 0.5 and range < 0.1. R-b if all O1 < 0.3. R-c if units template = circle in >= 70% of cells.

### H2: "Carries are events" (hierarchical discontinuities, nonlinear successor)
Motivation. LLMs fail arithmetic mostly at carries. Geometry papers ignore carries. A smooth number line would have no
special places; a base-10 tree would have discontinuities at 10, 100, 1000. My block-holdout failure (7xx cannot be
interpolated from the others) looks tree-like.
Claim. (H2a) Representation distance between n and n+1 has a hierarchy of spikes at carries (n ending in 9, 99, 999),
beyond what the additive digit model predicts. (H2b) Within a decade there is a LINEAR successor operator T
(z(n+1) ~ T z(n)); it breaks at carries, and the breakage grows with carry depth. (H2c) Coarse geometry is closer to a
prefix TREE (ultrametric) than to a line; a line appears only inside a leading-digit group.
Rivals. Pure smooth line (no spikes, one global T); pure circle/Fourier code (T exists everywhere as a rotation);
pure additive categorical code (spikes are fully explained by the digit model, no extra carry term).
If right: "successor is linear except at carries" is a crisp mechanistic claim that also links geometry to arithmetic
errors (Stage 5). If wrong: a global linear successor operator would support the Fourier/helix view, which is also publishable.
Method.
 1. Step length s(n) = ||z(n+1) - z(n)||. Fit s(n) = smooth trend + sum_j a_j * 1[carry level >= j] + additive-model
    prediction. Test a_j > 0 and a_2 > a_1 > 0 (carry level j = number of trailing 9s).
 2. Successor operator: fit ridge T on steps with carry level 0 (training numbers only). Report held-out R2 for: new
    carry-level-0 steps, carry-level-1, level-2, level-3 steps. Hierarchy = R2 drops with level.
 3. Tree vs line. For all pairs, fit E||z_m - z_n||^2 with: (a) a monotone function of |m-n| (isotonic), (b) a tree model
    = sum over prefix levels of weights x 1[prefixes differ at that level], (c) both. Cross-validate on held-out
    numbers AND on held-out whole leading-digit groups. Also report the ultrametric-violation index and Gromov delta
    (normalised by diameter) of the prefix-group centroids, and the isotonic (line) R2 INSIDE each leading-digit group.
Controls. Shuffle which n is called a carry; planted smooth curve with and without planted carry jumps at real noise;
matched-smoothness fake.
Decision rule. H2a: a_1 > 0, a_2 > a_1, CIs exclude 0, >= 70% of cells, >= 2 tokenizer families. H2b: R2(level 0) -
R2(level >= 1) > 0.20. H2c: tree beats line on held-out GROUPS by >= 0.10 R2 in late layers of digit-splitting models;
report that the line wins (if it does) in Pythia mid layers.

### H0: "Is it the number or the copy?" (invariance)
See section 1. Claim: the H1 and H2 structure is present when the number is READ (not only when it is about to be copied),
in other tasks, and in the static embeddings.
Method (needs new model runs, ~1-2 h per model on a laptop for 5 prompts x 10,000 numbers; start with Pythia-1.4B and Qwen2.5-1.5B).
 1. Read positions: R1 = last token of n (before "="); R2 = the "=" (current data); R3 = mean over n's tokens.
 2. Task contexts (5 paraphrases each): copy (current); quantity sentence ("She bought n apples. Then"); comparison
    ("Which is larger: n or 500? Answer:"); successor ("n + 1 ="); neutral list ("3, 8, n,").
 3. Static: for single-token numbers, rows of the input embedding E(n) and the unembedding U(n).
 4. Compute the "structure panel" on each: held-out R2 of smooth f(n); additive digit R2; block-holdout R2; H1 metrics
    (O1 by position, templates); H2 metrics (carry spikes, successor R2 by level, tree vs line).
 5. Noise ceiling: split-half reliability across paraphrases per number -> report every R2 as a FRACTION of the ceiling.
 6. FIX THE POSITIVE CONTROL: for days and months use an Engels-style setup (e.g. "Today is Monday. In 3 days it is"),
    project onto the top-k PCs first, then score circular order. The detector must find the days/months circle before any
    topology result is trusted again. If it cannot, drop topology claims from the paper.
Decision rule. Invariance holds if the panel metrics on R1 and on >= 3 of 5 contexts are >= 50% of the R2 values
and have the same sign/ordering of effects. If not, the paper becomes "number structure is largely output geometry"
(then report which parts survive).

### H3: "Quantity, composition, lexicon" (three-part decomposition; the leftover is not noise)
Motivation. E12b/E12c leave 20-70% of variance unexplained, which killed every dimension test. Nobody knows if that is
noise, per-number idiosyncrasy, or more structure. Also: my E10/E11 found that frequent numbers sit slightly further apart,
mostly through vector LENGTH, and rare numbers are slightly crowded. That fits a "lexical" component that frequent
numbers have more of (years, round numbers, 100, 1000, 2020, 1971).
Claim. z(n) = magnitude scaffold f(n) + compositional digit code D(n) + lexical component L(n). L(n) is REPRODUCIBLE
across paraphrases (not noise), grows with numeral frequency and "special" numbers, is model-specific, and accounts for
the frequency effect on vector length and crowding.
Rivals. Residual is pure noise (reliability ~ 0): then numbers are fully "magnitude + composition", a clean and strong result.
Method. (needs the paraphrases from H0)
 1. Remove f and D (fitted on training numbers). Residual vector r(n). Reliability = correlation of r(n) between two
    halves of the paraphrases (per number, then averaged), after removing the shared mean.
 2. Regress ||r(n)|| and the vector length on log SlimPajama frequency (partial out digit length), and on flags:
    year 1900-2030, multiple of 100/1000, repdigit, power of 2, famous numbers (007, 404, 911, 666, 1337).
 3. Re-run my E10/E11 crowding analysis with r(n) REMOVED. Does the rare/frequent crowding effect disappear?
 4. Across the 25 DataDecide recipes: for the same number, does ||r(n)|| rise with that recipe's count of n? (counts vary
    only ~1.5x, so low power; say so.)
Decision rule. Lexicon exists if reliability >= 0.30 and the frequency slope CI excludes 0 in >= 2 tokenizer families.
Crowding is "lexical" if its effect size falls by >= 50% after removing r(n).

### H5: "Double dissociation" (the model actually uses these parts)  [needs H1/H2 to pass first]
Motivation. Describing a shape is a weak claim; showing the model uses it is what top venues want.
Claim. Three subspaces are separately used: the magnitude subspace S_mag, the digit-code subspaces S_dig(p), and a carry
subspace S_carry. Editing S_mag changes magnitude judgements but not digit identity; editing S_dig changes digit-level
answers but not magnitude judgements; editing S_carry changes arithmetic only on carry items.
Rivals. Editing any of them hurts everything equally (entangled code), or nothing (epiphenomenal geometry).
Method. Models: Pythia-1.4B, Qwen2.5-1.5B, OLMo-2-1B.
 1. Subspaces from the training split at a chosen layer (chosen on a dev split, never on test items):
    S_mag = top-m directions of the fitted f(n) (m by CV, ~3-5); S_dig(p) = span of the 9 centroid differences of position p;
    S_carry = span of the carry effects from H2. Controls: random subspaces of equal dimension and orthogonal to the others;
    subspaces fitted on shuffled n.
 2. Interventions at number-token positions: (i) project-out ablation; (ii) activation patching of only that subspace
    between a clean and a corrupted number; (iii) steering by + alpha * (f(n') - f(n)).
 3. Behaviours (logit-difference metrics, 500+ items each, balanced for carries and digit length):
    B1 which of two numbers is larger; B2 the last digit of n; B3 the leading digit of n; B4 n + 1; B5 a + b with and without carries;
    B6 copy n.
 4. Prediction matrix: S_mag -> B1 (B3 partly); S_dig(units) -> B2, B6; S_carry -> carry cases of B4/B5 only.
Decision rule. Target effect >= 2x the random-subspace effect AND off-target effect <= 1.2x random, with bootstrap CIs
over items, in >= 2 models. Report the full 3 x 6 effect matrix, including the failures.

### H6: "How it forms and what data controls" (training and frequency)  [do last]
Motivation. I already found the line forms abruptly (Pythia steps 2,000-4,000) and that frequency modulates spacing.
Now ask WHICH PART forms first and which part the data (not the tokenizer) controls. This ties to my alpha/beta paper.
Claim. (a) Emergence order: unit-digit categories -> leading-digit ordinality -> carry structure -> lexicon (frequency-dependent).
(b) With the tokenizer fixed (all 25 DataDecide models share one), numeral frequency alpha changes spacing (beta), the
size of the lexical component and the ordinality decay rate, but NOT the digit templates or the existence of carries.
Method. Compute the full panel (H1, H2, H3 metrics + comparison/addition accuracy) at all checkpoints; define the
emergence time = the step when a metric reaches 50% of its final value; test the predicted order with a permutation test
over metrics; compare with behaviour emergence times. For (b) regress each panel metric on recipe alpha across the 25 models
with recipe-level bootstrap. Optional, if compute allows: 4-6 small models (~160M) pretrained with numeral frequency
deliberately changed, as the causal data experiment.
Decision rule. Predicted order holds if the permutation p < 0.05 for the ordering of >= 3 metrics in both Pythia and OLMo-2.
(b) holds if beta, ordinality decay and lexical size vary with alpha (CI excludes 0) while template type and carry
spikes do not (slope CI includes 0 and is small).

(Side test already requested: R2-vs-k "knee" test of the smallest dimension that explains held-out numbers. Keep it, but
interpret it with the factorised picture: the expected dimension is about 1 for magnitude + ~9 per categorical digit
position, so a knee near those values supports H1.)

---------------------------------------------------------------------------------------------------
## 4. ORDER OF WORK AND GATES

STAGE 1 (existing vectors, no GPU, hours): H1 and H2 on all 9 models, 5 depths, sets of 2/3/4 digits.
  GATE 1: if neither H1 nor H2 passes -> write the workshop paper on the negative result and the method (power checks). Stop here.
STAGE 2 (new model runs): H0 invariance + fixed positive control + noise ceiling + paraphrase set for H3.
  GATE 2: if the panel mostly disappears on the input side -> reframe the paper as "output geometry"; keep H5 (causal) on the surviving parts.
STAGE 3: H3 lexicon (uses Stage 2 paraphrases).
STAGE 4: H5 causal dissociation (only if Stage 1 passed; define subspaces from Stage 1 and 3).
STAGE 5: H6 training and data.
STAGE 6: freeze everything, regenerate figures, one table of verdicts (supported / not supported / cannot tell) with numbers.

---------------------------------------------------------------------------------------------------
## 5. WHAT TO SEND BACK AFTER EACH STAGE

 - A one-page summary: question, what was run, the verdict for each hypothesis (supported / not supported / cannot tell) with
   the numbers, and the power-check result.
 - Figures: (S1) ordinality by digit position per model x depth; step length s(n) vs n with carry spikes marked;
   successor-operator R2 by carry level; tree-vs-line bars on held-out groups.
 - The tables (CSV) with all models, depths, sets, and controls.
 - Anything surprising and anything that contradicts my earlier results.
 - Commit hashes of the PREREG and the code, and a list of deviations.

---------------------------------------------------------------------------------------------------
## 6. WHAT EACH OUTCOME MEANS FOR THE PAPER

 - H1 + H2 + H5 all pass: headline paper, "How LLMs write numbers in their heads: a floating-point-like code with carry events,
   and the model uses it" (strongest, A* plausible).
 - H1 + H2 pass, H5 inconclusive: strong analysis paper; try a spotlight at a good conference; keep working on causal tests.
 - Everything fails but controls are solid: a methods-and-negative-results paper, "most manifold claims about number
   representations are not testable at realistic noise", with the power-check and positive-control framework. Workshop or
   the datasets-and-benchmarks style track.
 - H0 fails: "the number line in hidden states is mostly output geometry" (a corrective result; still publishable).

---------------------------------------------------------------------------------------------------
## 7. KNOWN RISKS (tell me if you see them materialising)

 - Estimators with no power (dimension, homology): do not use them for claims.
 - The additive digit model and the smooth f(n) can compete for the same variance in 2-digit sets; use the units-only variant too.
 - One prompt per number so far; Stage 2 fixes this.
 - Chunk-tokenizer sets are small after keeping only single-token numbers; say so and widen CIs.
 - Over-tuning: layers, subspace sizes and bandwidths must be chosen on a dev split, never on the test items.

---------------------------------------------------------------------------------------------------
## 8. PRACTICAL

 - Put new work in `experiments/E13_number_code/` (README, PREREG_*.md per stage, src/, results/).
 - Reuse `src/e12_manifold.py`, `src/e12b_smooth_digits.py`, `src/e12c_heldout_r2.py` where possible.
 - Seeds fixed and recorded. Do not change seeds between a main run and a rerun (this caused trouble in the Q4 rerun).
 - Start now with STAGE 1 only. Write PREREG_stage1.md first, commit it, then run, then report and wait.

---------------------------------------------------------------------------------------------------
## 9. BEFORE ANY NOVELTY CLAIM, SEARCH FOR (and tell me what you find)

 - Per-digit-position geometry of number representations (ordinal vs categorical vs circular).
 - "Successor" or "carry" geometry in LLM hidden states; carry-related features in arithmetic circuits.
 - Ultrametric / hierarchical / hyperbolic structure of number embeddings.
 - Lexical (frequency-dependent) vs compositional components of number representations.
 - Double dissociation / subspace patching for magnitude vs digit identity.
 - Closest known papers: Kantamneni & Tegmark (arXiv 2502.00873), Bassi & Tomar (arXiv 2607.04167), Levy & Geva, Fu et al.
   (arXiv 2604.20817), Karkada et al. (arXiv 2602.15029). Read them and list what they already did.
