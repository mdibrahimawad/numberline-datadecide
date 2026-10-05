# llm_natural_log_paperfaithful

This directory is a copy of `llm_natural_log` with the paper-described
Experiment 1 behavior added explicitly. The original directory is left
untouched.

## Main changes

- No empty `login(token="")`; HuggingFace auth is read from `HF_TOKEN` or
  `HUGGINGFACE_TOKEN` when present.
- Paper Eq. 4 intervals:
  - `G1 = {1, ..., 20}`
  - `Gi = {10^i - 19, ..., 10^i + 20}` for `i >= 2`
- Paper Eq. 5 context:
  - for target `x in Gi`, context examples `a,b,c` are sampled from the same
    group `Gi`, using `context="paper"`.
- Paper filtering:
  - for numerical prompts, keep a sample only if greedy generation after the
    final `=` produces the target number `x`.
  - This is controlled by `filter_correct=True` and `filter_data="numerics"`.
- Paper monotonicity reporting:
  - aggregate `mean(abs(rho))` and `std(abs(rho))`, avoiding PCA sign-flip
    cancellation across runs.
- Paper SRI beta:
  - fit `min_{alpha,beta>0} sum_i (d_i - alpha beta^i)^2` directly via
    geometric regression.
  - The released repo's previous log-linear approximation is still available
    as `beta_method="repo_loglinear"`.
- Letter control:
  - `letter_mode="paper_token_length"` chooses random-letter sequence lengths
    from tokenizer lengths of representative powers of ten.

## Run

Install dependencies in the active environment:

```bash
pip install -r requirements.txt
```

Run a Table 1 style job:

```bash
python Table_1.py pythia
```

or with an exact HuggingFace model id:

```bash
python Table_1.py EleutherAI/pythia-2.8b
```

The defaults are the paper-faithful settings:

```text
context=paper
interval_mode=paper
filter_correct=True
filter_data=numerics
beta_method=geometric
letter_mode=paper_token_length
```

The code writes the same `ICLR_results/table_1_PCA.txt` and
`ICLR_results/table_1_PLS.txt` files as the released repo.
