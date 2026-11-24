# E15 pre-registration: does token count make the log number line? (written before any real data is run)

**Question (from the Veli / Hilal discussion, 2026-11-23).** In one-hot digit tables, PCA gives a log-like line because
PC1 counts the blank leading slots (digit length). Hilal's proposal for LLMs: at the final `=`, the hidden state is an
attention-weighted mix of the query number's tokens; each extra token adds a roughly constant push, while each extra
digit multiplies the value by 10, so equal pushes per ×10 look logarithmic. If so, the "leading zeros" of the LLM are its
**token count**, and the log pattern should follow the tokenizer.

E14 found length explains a median 28 % of the log pattern in the DataDecide models, with the jump at 99→100 where the
NeoX tokenizer adds no token. E15 tests the mechanism directly with inputs that change length without changing value.

## Models (fixed)
| Model | Tokenizer type | Token count in `,n=` (checked before any hidden state was computed) |
|---|---|---|
| EleutherAI/pythia-1.4b | NeoX chunks; **identical to DataDecide's tokenizer on n = 0…9999 (10,000/10,000)** | 1–99: 1; 100–999: 1 (777) or 2 (123); 1000–9999: 2 (8,861) or 1 (139) |
| Qwen/Qwen2.5-1.5B | one token per digit | = number of digits |
| allenai/OLMo-2-0425-1B | groups of up to 3 digits (as Llama 3) | 1–999: 1; 1000–9999: 2 |

fp32 on CPU, final checkpoints, no fine-tuning.

## Prompts and vectors (fixed)
- Prompt: `a=a,b=b,c=c,x=` with a, b, c uniform integers in [0, 10000] (the E06 / β format). `x` is the query string
  (possibly formatted, below).
- 3 context draws per item (seeds fixed per item); the hidden state of the final token `=` at every layer is averaged
  over the 3 draws → one vector per item and layer.
- No correct-output filter (the manipulated strings may not be copied). Copy accuracy (greedy first token = first token
  of `x`) is recorded as a diagnostic.

## Number axis (fixed, fitted on baseline numbers only)
- **Baseline set B:** all 1–99, 300 random 100–999 and 300 random 1000–9999 (seed 0), plus every number in the
  boundary blocks 90–109 and 990–1009. Plain formatting.
- Per layer: PCA on the centred baseline vectors; PC1 oriented so that its correlation with log10 n is positive;
  calibration PC1 = a + b·log10 n by least squares on B. **Position u(x) = (PC1(x) − a) / b**, in decades
  (1 unit = the axis distance of one ×10 step). Test items are projected on the baseline PC1 and use the same a, b.
- **Primary layer L\***: argmax over layers ≥ 1 of √(EV₁ · |Spearman(PC1, n)|) on B (the project's joint rule).
  All layers are reported; verdicts use L\* only.

## Tests and pass rules (primary, at L\*)
CIs are 95 % bootstrap over items (2,000 resamples).

**T1 Leading zeros (Veli's question).** Items: n = 1…9 written with width 1, 2, 3, 4 (`7`, `07`, `007`, `0007`); n = 10…99
with width 2, 3, 4 (`42`, `042`, `0042`). Shift per added zero: R_pad = mean over items and k of [u(padded) − u(plain)] / k,
k = number of added zeros.
- Value account (zeros ignored): R_pad ≈ 0. Pure length account: R_pad ≈ 1 (one decade per extra slot).
- Verdict: R_pad CI entirely < 0.2 → **"value"**; CI entirely > 0.5 → **"length"**; otherwise **"partial"**.
- Token-count prediction (Hilal): in Qwen each zero is an extra token; in Pythia `07`, `007`, `0007` are single tokens;
  in OLMo-2 `07`, `007` are single tokens and `0007` is two. If token count (not characters) drives position, R_pad is
  large in Qwen and ≈ 0 in Pythia (and OLMo-2 for widths ≤ 3). Reported as a pattern across models.

**T2 Attention to the query (Hilal's mechanism; descriptive).** For baseline items, at every layer: the share of the
final `=`'s attention (averaged over heads) on the query number's tokens, and how it changes with the query's token count
(Spearman across items). The mechanism needs the `=` to attend substantially to the query tokens in layers ≤ L\*.
Reported, no pass rule (attention weights alone cannot show the push per token is constant).

**T3 Same value, more tokens.** 100 random numbers 1000–9999 (seed 1) written plain (`1234`), with a space (`1 234`) and
with an underscore (`1_234`). (Commas are not used: they separate the demonstrations in the prompt.) Shift
Δ_f = mean u(format) − u(plain), and tokens added per format per model.
- Value account: |Δ| < 0.2 decades. Token account: Δ > 0 and growing with the number of added tokens.
- Verdict per model and format: CI entirely within (−0.2, 0.2) → **"value"**; CI entirely > 0.2 → **"tokens push up"**;
  CI entirely < −0.2 → **"pushes down"**; otherwise **"unclear"**.

**T4 Where the jump sits (tokenizer comparison).** On baseline, jump at boundary b ∈ {100, 1000}:
J_b = mean u(b…b+9) − mean u(b−10…b−1) (the smooth log change across these blocks is < 0.05 decades).
- Token-count changes at the boundary: Qwen +1 at 100 and +1 at 1000; OLMo-2 0 at 100, +1 at 1000; Pythia 0 at 100,
  ≈ +1 at 1000 (counted on the actual items and reported).
- Token account predicts: jump (J_b CI > 0.1) where a token is added, none (CI includes 0 or J_b < 0.1) where not.
- Verdict per model: **"follows tokens"** if both boundaries match; **"does not follow tokens"** if neither matches;
  **"mixed"** otherwise.

## Overall reading (fixed now)
- Hilal's mechanism is **supported** if, across models, T1/T3 shifts and T4 jumps go with added tokens (not added
  characters or digits).
- It is **contradicted** if shifts or jumps appear where no token is added (e.g. a Pythia jump at 99→100, or Pythia
  T1 "length" although `0007` is one token).

## Not tested / caveats
- Attention weights are not contributions (T2 is descriptive only). One prompt format. Three models.
- Pythia stands in for the DataDecide models (same tokenization, different weights and data).
