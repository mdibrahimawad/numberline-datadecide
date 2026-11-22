# E14: is the log number line just digit length?

**Question.** Prof. Hilal showed that one-hot digit tables (no model) give a PC1 that is linear in
log10(n), because PC1 counts the blank digit slots. Is that also why LLMs, and our β, show a log
number line?

**Idea.** A digit-length staircase and a true log line space the decades the same way. They differ
only at the length boundaries (99→100, 999→1000) and inside a length (100→999). We measure:
- the jump at the boundary as a share of a decade's rise (LS);
- the shape inside a length (λ: 0 = log, 1 = linear).

**Data.** Existing vectors only, so no GPU:
- E06 β data (25 DataDecide + 6 Paloma models);
- E13 vectors (Pythia, Qwen; every number 10–9999; copy and sentence readings).

**Pre-registration.** [PREREG.md](PREREG.md), frozen before the real-data run.
**Code.** `src/e14_length_log.py`, `src/e14_report.py`, `tests/test_e14.py`.

**How to run.**
```bash
python -m src.e14_length_log power --out results/e14      # synthetic power check
python -m src.e14_length_log e06   --out results/e14      # D1 + D2 (E06 .npz in the repo)
python -m src.e14_length_log e13   --data <dir with pythia-1.4b/, qwen2.5-1.5b/ from mdibrahimawad/e13-number-code> --out results/e14
python -m src.e14_length_log report --out results/e14     # -> results/e14/RESULTS.md
```

**Status.** Done (2026-11-22). Pre-registration frozen at commit 14c3973; no deviations. Full tables: `results/e14/RESULTS.md`.

**Result (pre-registered verdict: PARTLY RIGHT).**

| Data | Verdict / class | Median LS (share of a decade's rise that is a length jump) | Shape inside a length |
|---|---|---|---|
| 25 DataDecide, β layer | partly right: 21 mixed, 3 smooth-log, 1 smooth, 0 staircase | 0.28 (jump at 100: 0.55; at 1000: 0.13) | log-shaped in 22/25 (λ CI < 0.5), linear in 1/25 |
| 25 DataDecide, layer 8 | partly right: 13 mixed, 12 smooth-log | 0.21 | median λ −0.25 |
| 6 Paloma | partly right (3 mixed, 1 smooth-log, 1 smooth, 1 can't tell) | 0.18 | — |
| Pythia, copy / sentence | mixed / mixed | 0.23 / 0.31 (at 100: 0.40/0.44; at 1000: 0.07/0.18) | log-like (λ 0.05 / 0.25) |
| Qwen (digit tokenizer), copy | **length staircase** | 0.69 (at 100: 0.85; at 1000: 0.53) | unclear |
| Qwen, sentence | smooth, log inside | 0.18 | log (λ −0.40) |

- Digit length explains roughly a quarter of the log pattern in the β models. The rest is a smooth log line,
  and the spacing *inside* one digit length is itself log-shaped, which digit length cannot produce.
- In the NeoX-tokenizer models, the jump sits at 99→100 (no change in token count) and mostly not at 999→1000
  (where the token count changes). So the length effect is not the input format. Neither pre-registered tokenizer
  pattern was met (0/25 each), because the step at 1000 is small.
- The professor's mechanism does hold for the digit-by-digit tokenizer in the copy reading (Qwen: staircase).
- Late DataDecide layers (12–16) lose the line (ρ < 0.9); not used for verdicts.

