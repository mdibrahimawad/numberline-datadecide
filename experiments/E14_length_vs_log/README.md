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

**Status / result.** See `results/e14/RESULTS.md`.
