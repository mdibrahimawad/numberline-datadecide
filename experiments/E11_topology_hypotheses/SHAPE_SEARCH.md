# E11 shape search: find number sets with topological shape, then test them on held-out models (pre-registered 2026-10-16)

**Written before the search was run.** Idea (owner's): search freely ("overfit") for sets of numbers whose points
form a topological shape in one model, lock what is found, and test it on models that took no part in the search.
Code: `src/e11_shape_search.py` (`--phase discover`, then `--phase test` on the locked list).

## Models
| Role | Digit-splitting | Chunk tokenizers |
|---|---|---|
| Discovery | Crystal | Pythia-1.4B |
| Held-out test | OpenLLaMA-3B, StarCoderBase-3B, Qwen2.5-1.5B, SmolLM2-1.7B | OLMo-2-1B (+ RedPajama-3B and Llama-3.2-1B when extracted; reported) |
All five relative depths (~25 / 50 / 75 / 84 % / last). These models were used in E10/E11 for other questions; none
of the sets or shape scores below was computed on them before.

## Candidate number sets (fixed list, 48)
- **Point sets** (each number one point): ranges 1–9, 1–12, 1–24, 1–31, 1–59, 1–100, 10–99, 100–199, 1900–2025;
  multiples of 10, 25, 50, 100, 1000 (in 1–10000); squares; primes; powers of 2; repdigits (11…99, 111…999, 1111…9999).
  Sets above 300 points are subsampled to 300 (fixed seed).
- **Class-centroid sets** (one point per class = mean of its members after removing the mean of the higher digits'
  group): n mod k for k = 4, 5, 6, 7, 8, 9, 11, 12, 20, 24, 25, 60, 100, separately over 1–999 and over
  1000–9999 (no 1500–2099); and the units, tens, hundreds and thousands digit of 1000–9999 (10 classes each).

## Shape scores (all on distances divided by the set's median distance)
- **S0 clusters**: mean of the 3 largest H0 deaths ÷ median H0 death.
- **S1 loop**: lifetime of the longest H1 bar.
- **S2 void**: lifetime of the longest H2 bar (sets of ≤ 100 points only).
- **S1b second loop**: lifetime of the second-longest H1 bar (two loops: figure-8 or, with S2, a torus).
- Centroid sets only (geometry, not topology; topology cannot tell a polygon from a circle):
  - **Spoly polygon in order**: share of classes whose two nearest classes are their cyclic neighbours k − 1 and
    k + 1 (mod the number of classes), set to 0 unless the first and last class are also neighbours (the loop
    closes; a line does not). 1 = a regular polygon/clock in numeric order.
  - **Sline line in order**: |Spearman(class value, first principal component)| × that component's variance share.

Shape names used in the report (from which scores are significant): clusters (S0), loop/polygon (S1, and Spoly if
in numeric order), figure-8 (S1 + S1b), sphere (S2), torus (S1 + S1b + S2), ordered line (Sline).

## Nulls
- Point sets: 100 (discovery) / 1,000 (test) Gaussian clouds with the same size and covariance spectrum as the set.
- Centroid sets: the same number of class-label shuffles within the population.
p = (#null ≥ observed + 1) / (#null + 1), one-sided (more structure than the null); z = (obs − null mean) / null sd.

## Locking rule (applied automatically to the discovery output)
For each discovery model separately: all (set, score, depth) with p = 1/101 (beaten by no null), ranked by z; the
**top 10** are locked, with their depth. The locked list is committed before the test phase runs.

## Test rule
Each locked item is tested at its locked depth with 1,000 nulls, in the held-out models of the **same tokenizer type**
as its discovery model. Significance threshold: Bonferroni p < 0.05 / 10 = 0.005.
- Crystal items: **replicate** if significant in ≥ 3 of the 4 held-out digit-splitting models.
- Pythia items: replicate if significant in OLMo-2 and in ≥ 1 of RedPajama-3B / Llama-3.2-1B once extracted;
  until then reported as "OLMo-2 only".
Every locked item is reported with its result, passed or not.


**Implementation note (2026-10-16, before any result was seen):** the first discovery run hit the 2-hour job limit
before writing anything. The script now saves each (model, depth, set) cell as it finishes and resumes from them,
and uses one thread per worker. Sets, scores, nulls, seeds and the locking/test rules are unchanged.

## Results (run 2026-10-16, exactly as above; locked list committed in `8156842` before the test)
**Discovery** (`results/e11_shape_search/discovery.csv`, 48 sets × 6 scores × 5 depths × 2 models): the top 10
per model are all **S0 clusters among the class centroids of n mod 6, 12, 24 and 60 over 1000–9999** (z 90–224).
**Test** (`test.csv`, 1,000 nulls): **all 20 locked items significant in every held-out model** (p = 0.001,
z 14–343): Crystal's 10 in OpenLLaMA, StarCoder, Qwen2.5 and SmolLM2 (4/4); Pythia's 10 in OLMo-2, RedPajama and
Llama-3.2 (3/3). **All replicate.**

**What the clusters are (check after the test):** the centroids of mod 6 and mod 12 split into exactly the even and
the odd classes (single-linkage, 2 clusters: ARI with parity 1.00 in Crystal, Qwen2.5, OLMo-2, RedPajama); the 60
centroids of mod 60 split into 10 clusters by last digit (ARI 0.89–1.00). The replicated shape is the models'
**last-digit (and hence parity) code** seen through residue classes, not knowledge of mod 6/12/24/60: moduli coprime
with 10 (7, 9, 11) give no clusters over 1000–9999 (z ≤ 5). Consistent with period-2/5/10 number features (Fu et al.
2026); not new.

**Leads not locked (exploratory, for a separate pre-registration):** Pythia, numbers 1–999 (single tokens): clusters for
mod 7, 9, 11 (z 22–41) and an ordered line for mod 11 (z ≈ 12), absent in Crystal; mod-100 endings: short loops and
ordered lines (z 12–37); clock-like polygons in numeric order: 1 hit in 480 cells.
