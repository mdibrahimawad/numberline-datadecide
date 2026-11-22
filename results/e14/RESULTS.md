# E14 results: is the log number line just digit length?

Pre-registration: `experiments/E14_length_vs_log/PREREG.md`. LS = share of each decade's rise that happens as a jump at the digit-length boundary (0 = smooth log line, 1 = pure length staircase). λ = shape inside a length (0 = log, 1 = linear). 95 % bootstrap CIs in brackets.

## Summary

| data | layer / reading | verdict / class | median LS | n |
|---|---|---|---|---|
| E06_DataDecide | primary: E06-selected layer | PARTLY RIGHT (mixed) | +0.28 | 25 |
| E06_DataDecide | secondary: fixed layer 8 | PARTLY RIGHT (mixed) | +0.21 | 25 |
| E06_Paloma | primary: E06-selected layer | PARTLY RIGHT (mixed) | +0.18 | 6 |
| E06_Paloma | secondary: fixed layer 8 | PARTLY RIGHT (mixed) | +0.32 | 6 |
| E13 pythia-1.4b | copy '=' (C1, beta-like) | mixed | +0.23 | 1 |
| E13 pythia-1.4b | sentence (AVG5, number token) | mixed | +0.31 | 1 |
| E13 qwen2.5-1.5b | copy '=' (C1, beta-like) | length staircase | +0.69 | 1 |
| E13 qwen2.5-1.5b | sentence (AVG5, number token) | smooth, log inside | +0.18 | 1 |

## D1: 25 DataDecide models (β protocol) — main verdict — primary: E06-selected layer

**Verdict: PARTLY RIGHT (mixed).** Classes: {'mixed': 21, 'smooth, log inside': 3, 'smooth, shape unclear': 1}. Median LS = +0.28 (LS at 100: +0.55, at 1000: +0.13); median λ = -0.13.

Tokenizer sub-test: input-format pattern (no step at 100, step at 1000) in 0/25; learned-length pattern (step at both) in 0/25.

| model | layer | LS | LS at 100 | LS at 1000 | J | λ inside | ρ | class |
|---|---|---|---|---|---|---|---|---|
| c4 | 3 | +0.07 [+0.03, +0.12] | +0.62 [+0.61, +0.64] | -0.47 [-0.56, -0.40] | +0.40 | +0.05 [-0.05, +0.17] | 0.93 | smooth, log inside |
| dclm-baseline-25p-dolma1.7-75p | 8 | +0.36 [+0.32, +0.39] | +0.31 [+0.25, +0.35] | +0.41 [+0.38, +0.45] | +0.23 | +0.33 [+0.18, +0.50] | 0.94 | mixed |
| dclm-baseline-50p-dolma1.7-50p | 8 | +0.23 [+0.20, +0.27] | +0.39 [+0.36, +0.42] | +0.07 [+0.03, +0.12] | +0.18 | +0.02 [-0.07, +0.15] | 0.95 | mixed |
| dclm-baseline-75p-dolma1.7-25p | 10 | +0.26 [+0.22, +0.29] | +0.48 [+0.45, +0.50] | +0.03 [-0.02, +0.09] | +0.27 | +0.00 [-0.10, +0.10] | 0.95 | mixed |
| dclm-baseline-qc-10p | 8 | +0.20 [+0.15, +0.24] | +0.54 [+0.52, +0.57] | -0.14 [-0.21, -0.07] | +0.25 | -0.13 [-0.28, +0.03] | 0.94 | smooth, log inside |
| dclm-baseline-qc-20p | 9 | +0.36 [+0.33, +0.39] | +0.59 [+0.57, +0.61] | +0.13 [+0.08, +0.17] | +0.42 | -0.13 [-0.28, -0.02] | 0.96 | mixed |
| dclm-baseline-qc-7p-fw2 | 8 | +0.28 [+0.25, +0.31] | +0.59 [+0.56, +0.61] | -0.02 [-0.08, +0.02] | +0.34 | -1.00 [-1.00, -0.88] | 0.95 | mixed |
| dclm-baseline-qc-7p-fw3 | 3 | +0.49 [+0.47, +0.52] | +0.55 [+0.53, +0.57] | +0.43 [+0.40, +0.46] | +0.33 | -0.07 [-0.25, +0.12] | 0.95 | mixed |
| dclm-baseline-qc-fw-10p | 3 | +0.29 [+0.26, +0.31] | +0.47 [+0.45, +0.49] | +0.10 [+0.06, +0.14] | +0.26 | -0.22 [-0.32, -0.13] | 0.96 | mixed |
| dclm-baseline-qc-fw-3p | 2 | +0.19 [+0.14, +0.24] | +0.12 [+0.08, +0.16] | +0.26 [+0.19, +0.32] | +0.16 | +0.52 [+0.40, +0.65] | 0.94 | smooth, shape unclear |
| dclm-baseline | 8 | +0.27 [+0.22, +0.33] | +0.65 [+0.63, +0.68] | -0.11 [-0.19, -0.02] | +0.45 | -0.13 [-0.30, +0.05] | 0.92 | mixed |
| dolma1_6plus | 9 | +0.43 [+0.39, +0.47] | +0.41 [+0.36, +0.45] | +0.45 [+0.40, +0.49] | +0.30 | -0.35 [-0.57, -0.11] | 0.93 | mixed |
| dolma1_7-no-code | 8 | +0.38 [+0.34, +0.41] | +0.59 [+0.56, +0.61] | +0.17 [+0.12, +0.22] | +0.40 | -0.02 [-0.15, +0.12] | 0.95 | mixed |
| dolma1_7-no-flan | 9 | +0.40 [+0.37, +0.45] | +0.30 [+0.25, +0.36] | +0.51 [+0.48, +0.54] | +0.27 | +0.20 [+0.03, +0.37] | 0.94 | mixed |
| dolma1_7-no-math-code | 4 | -0.09 [-0.17, -0.02] | +0.68 [+0.67, +0.69] | -0.85 [-1.01, -0.73] | +0.54 | -0.60 [-0.73, -0.48] | 0.92 | smooth, log inside |
| dolma1_7-no-reddit | 8 | +0.35 [+0.30, +0.40] | +0.24 [+0.17, +0.30] | +0.46 [+0.43, +0.50] | +0.29 | -0.25 [-0.47, -0.06] | 0.93 | mixed |
| dolma1_7 | 4 | +0.27 [+0.25, +0.29] | +0.40 [+0.38, +0.41] | +0.14 [+0.12, +0.17] | +0.22 | -0.07 [-0.13, +0.00] | 0.98 | mixed |
| falcon-and-cc-qc-10p | 3 | +0.28 [+0.25, +0.30] | +0.35 [+0.33, +0.37] | +0.20 [+0.16, +0.24] | +0.24 | +0.13 [+0.05, +0.20] | 0.97 | mixed |
| falcon-and-cc-qc-20p | 8 | +0.29 [+0.25, +0.34] | +0.61 [+0.59, +0.63] | -0.02 [-0.09, +0.04] | +0.32 | -0.17 [-0.38, +0.03] | 0.94 | mixed |
| falcon-and-cc-qc-orig-10p | 3 | +0.26 [+0.24, +0.29] | +0.55 [+0.54, +0.56] | -0.03 [-0.06, +0.01] | +0.36 | -0.35 [-0.43, -0.27] | 0.97 | mixed |
| falcon-and-cc-qc-tulu-10p | 2 | +0.24 [+0.17, +0.29] | +0.00 [-0.07, +0.07] | +0.47 [+0.41, +0.52] | +0.27 | +1.28 [+1.10, +1.48] | 0.91 | mixed |
| falcon-and-cc | 9 | +0.35 [+0.31, +0.40] | +0.71 [+0.69, +0.73] | -0.00 [-0.09, +0.08] | +0.55 | -0.20 [-0.38, -0.04] | 0.93 | mixed |
| falcon | 8 | +0.39 [+0.35, +0.43] | +0.65 [+0.63, +0.67] | +0.13 [+0.07, +0.19] | +0.46 | -0.90 [-0.98, -0.74] | 0.94 | mixed |
| fineweb-edu | 9 | +0.48 [+0.45, +0.50] | +0.56 [+0.53, +0.58] | +0.39 [+0.36, +0.43] | +0.46 | -0.38 [-0.53, -0.28] | 0.96 | mixed |
| fineweb-pro | 7 | +0.21 [+0.15, +0.26] | +0.73 [+0.72, +0.75] | -0.32 [-0.42, -0.22] | +0.52 | -0.17 [-0.33, +0.02] | 0.93 | mixed |

## D1: 25 DataDecide models (β protocol) — main verdict — secondary: fixed layer 8

**Verdict: PARTLY RIGHT (mixed).** Classes: {'mixed': 13, 'smooth, log inside': 12}. Median LS = +0.21 (LS at 100: +0.54, at 1000: -0.11); median λ = -0.25.

Tokenizer sub-test: input-format pattern (no step at 100, step at 1000) in 0/25; learned-length pattern (step at both) in 0/25.

| model | layer | LS | LS at 100 | LS at 1000 | J | λ inside | ρ | class |
|---|---|---|---|---|---|---|---|---|
| c4 | 8 | -0.39 [-0.51, -0.29] | +0.63 [+0.62, +0.65] | -1.42 [-1.65, -1.24] | +0.34 | -0.63 [-0.80, -0.48] | 0.90 | smooth, log inside |
| dclm-baseline-25p-dolma1.7-75p | 8 | +0.36 [+0.32, +0.39] | +0.31 [+0.25, +0.35] | +0.41 [+0.38, +0.45] | +0.23 | +0.33 [+0.18, +0.50] | 0.94 | mixed |
| dclm-baseline-50p-dolma1.7-50p | 8 | +0.23 [+0.20, +0.27] | +0.39 [+0.36, +0.42] | +0.07 [+0.03, +0.12] | +0.18 | +0.02 [-0.07, +0.15] | 0.95 | mixed |
| dclm-baseline-75p-dolma1.7-25p | 8 | +0.21 [+0.18, +0.24] | +0.47 [+0.45, +0.49] | -0.04 [-0.09, +0.01] | +0.25 | -0.05 [-0.15, +0.03] | 0.96 | smooth, log inside |
| dclm-baseline-qc-10p | 8 | +0.20 [+0.15, +0.24] | +0.54 [+0.52, +0.57] | -0.14 [-0.21, -0.07] | +0.25 | -0.13 [-0.28, +0.03] | 0.94 | smooth, log inside |
| dclm-baseline-qc-20p | 8 | -0.10 [-0.16, -0.04] | +0.52 [+0.50, +0.54] | -0.72 [-0.82, -0.61] | +0.31 | -0.45 [-0.57, -0.35] | 0.95 | smooth, log inside |
| dclm-baseline-qc-7p-fw2 | 8 | +0.28 [+0.25, +0.31] | +0.59 [+0.56, +0.61] | -0.02 [-0.08, +0.02] | +0.34 | -1.00 [-1.00, -0.88] | 0.95 | mixed |
| dclm-baseline-qc-7p-fw3 | 8 | -0.39 [-0.49, -0.31] | +0.56 [+0.54, +0.57] | -1.33 [-1.52, -1.17] | +0.29 | -0.92 [-0.98, -0.83] | 0.90 | smooth, log inside |
| dclm-baseline-qc-fw-10p | 8 | -0.03 [-0.08, +0.02] | +0.48 [+0.46, +0.50] | -0.53 [-0.62, -0.45] | +0.18 | -0.02 [-0.12, +0.12] | 0.94 | smooth, log inside |
| dclm-baseline-qc-fw-3p | 8 | -0.52 [-0.62, -0.43] | +0.49 [+0.48, +0.52] | -1.53 [-1.73, -1.37] | +0.20 | -0.95 [-1.00, -0.88] | 0.90 | smooth, log inside |
| dclm-baseline | 8 | +0.27 [+0.22, +0.33] | +0.65 [+0.63, +0.68] | -0.11 [-0.19, -0.02] | +0.45 | -0.13 [-0.30, +0.05] | 0.92 | mixed |
| dolma1_6plus | 8 | +0.29 [+0.25, +0.33] | +0.26 [+0.20, +0.30] | +0.32 [+0.28, +0.36] | +0.18 | +0.02 [-0.11, +0.17] | 0.95 | mixed |
| dolma1_7-no-code | 8 | +0.38 [+0.34, +0.41] | +0.59 [+0.56, +0.61] | +0.17 [+0.12, +0.22] | +0.40 | -0.02 [-0.15, +0.12] | 0.95 | mixed |
| dolma1_7-no-flan | 8 | +0.29 [+0.25, +0.33] | +0.29 [+0.25, +0.33] | +0.29 [+0.25, +0.33] | +0.18 | -0.13 [-0.27, -0.02] | 0.95 | mixed |
| dolma1_7-no-math-code | 8 | -0.06 [-0.12, -0.02] | +0.54 [+0.53, +0.56] | -0.67 [-0.77, -0.58] | +0.25 | -0.18 [-0.31, -0.08] | 0.94 | smooth, log inside |
| dolma1_7-no-reddit | 8 | +0.35 [+0.30, +0.40] | +0.24 [+0.17, +0.30] | +0.46 [+0.43, +0.50] | +0.29 | -0.25 [-0.47, -0.06] | 0.93 | mixed |
| dolma1_7 | 8 | +0.43 [+0.39, +0.47] | +0.52 [+0.48, +0.56] | +0.34 [+0.30, +0.39] | +0.36 | -0.22 [-0.43, -0.05] | 0.93 | mixed |
| falcon-and-cc-qc-10p | 8 | +0.10 [+0.03, +0.16] | +0.64 [+0.61, +0.66] | -0.44 [-0.57, -0.34] | +0.30 | -0.98 [-1.00, -0.83] | 0.91 | smooth, log inside |
| falcon-and-cc-qc-20p | 8 | +0.29 [+0.25, +0.34] | +0.61 [+0.59, +0.63] | -0.02 [-0.09, +0.04] | +0.32 | -0.17 [-0.38, +0.03] | 0.94 | mixed |
| falcon-and-cc-qc-orig-10p | 8 | +0.19 [+0.14, +0.23] | +0.57 [+0.54, +0.59] | -0.18 [-0.26, -0.12] | +0.28 | -0.47 [-0.62, -0.30] | 0.94 | smooth, log inside |
| falcon-and-cc-qc-tulu-10p | 8 | +0.21 [+0.16, +0.25] | +0.62 [+0.60, +0.64] | -0.21 [-0.29, -0.13] | +0.40 | -0.62 [-0.78, -0.43] | 0.94 | smooth, log inside |
| falcon-and-cc | 8 | +0.25 [+0.19, +0.31] | +0.72 [+0.70, +0.74] | -0.22 [-0.33, -0.12] | +0.56 | -0.25 [-0.41, -0.08] | 0.92 | mixed |
| falcon | 8 | +0.39 [+0.35, +0.43] | +0.65 [+0.63, +0.67] | +0.13 [+0.07, +0.19] | +0.46 | -0.90 [-0.98, -0.74] | 0.94 | mixed |
| fineweb-edu | 8 | +0.21 [+0.17, +0.26] | +0.54 [+0.52, +0.57] | -0.11 [-0.20, -0.03] | +0.39 | -0.75 [-0.87, -0.62] | 0.92 | mixed |
| fineweb-pro | 8 | -0.23 [-0.41, -0.11] | +0.68 [+0.66, +0.70] | -1.13 [-1.48, -0.91] | +0.48 | -0.47 [-0.67, -0.27] | 0.86 | smooth, log inside |

### E06_DataDecide: all layers (median over models, point estimates, exploratory)

| layer | LS | LS 100 | LS 1000 | J | λ | ρ |
|---|---|---|---|---|---|---|
| 1 | +0.75 | +0.37 | +1.15 | -0.94 | +0.08 | 0.07 |
| 2 | +0.00 | +0.03 | +0.08 | +0.15 | -0.02 | 0.90 |
| 3 | +0.18 | +0.41 | -0.05 | +0.20 | -0.23 | 0.96 |
| 4 | +0.18 | +0.48 | -0.01 | +0.30 | -0.25 | 0.96 |
| 5 | +0.13 | +0.54 | -0.21 | +0.33 | -0.35 | 0.96 |
| 6 | +0.12 | +0.53 | -0.34 | +0.29 | -0.30 | 0.95 |
| 7 | +0.20 | +0.54 | -0.21 | +0.29 | -0.40 | 0.95 |
| 8 | +0.21 | +0.54 | -0.11 | +0.30 | -0.25 | 0.94 |
| 9 | +0.27 | +0.56 | -0.00 | +0.32 | -0.25 | 0.94 |
| 10 | +0.33 | +0.57 | +0.05 | +0.34 | -0.15 | 0.94 |
| 11 | +0.26 | +0.54 | +0.04 | +0.33 | -0.13 | 0.94 |
| 12 | -0.36 | +0.04 | -1.14 | +0.06 | -0.18 | 0.89 |
| 13 | -3.22 | -0.92 | -7.47 | -1.04 | -0.20 | 0.44 |
| 14 | -8.84 | -4.44 | -12.44 | -4.54 | -0.58 | 0.30 |
| 15 | -4.91 | -2.09 | -26.57 | -2.89 | -0.95 | 0.33 |
| 16 | -2.61 | -0.40 | -9.55 | -1.38 | -0.58 | 0.56 |

## D2: 6 Paloma models (replication) — primary: E06-selected layer

**Verdict: PARTLY RIGHT (mixed).** Classes: {"can't tell": 1, 'mixed': 3, 'smooth, log inside': 1, 'smooth, shape unclear': 1}. Median LS = +0.18 (LS at 100: +0.40, at 1000: +0.36); median λ = +0.17.

Tokenizer sub-test: input-format pattern (no step at 100, step at 1000) in 0/6; learned-length pattern (step at both) in 0/6.

| model | layer | LS | LS at 100 | LS at 1000 | J | λ inside | ρ | class |
|---|---|---|---|---|---|---|---|---|
| paloma-c4 | 7 | -0.34 [-0.74, -0.14] | -1.15 [-1.92, -0.78] | +0.46 [+0.39, +0.53] | +0.15 | +1.87 [+1.57, +2.00] | 0.71 | can't tell |
| paloma-dolma | 1 | +0.37 [+0.21, +0.53] | +0.18 [-0.05, +0.39] | +0.56 [+0.44, +0.69] | +0.11 | +0.93 [+0.42, +1.53] | 0.53 | mixed |
| paloma-falcon-refinedweb | 5 | -0.09 [-0.17, -0.00] | +0.46 [+0.43, +0.49] | -0.63 [-0.80, -0.49] | +0.24 | -0.38 [-0.52, -0.27] | 0.88 | smooth, log inside |
| paloma-mc4 | 5 | +0.47 [+0.45, +0.50] | +0.42 [+0.39, +0.44] | +0.53 [+0.50, +0.55] | +0.42 | -0.13 [-0.26, -0.01] | 0.97 | mixed |
| paloma-pile | 5 | +0.33 [+0.30, +0.36] | +0.39 [+0.37, +0.42] | +0.26 [+0.23, +0.30] | +0.21 | -0.30 [-0.42, -0.17] | 0.96 | mixed |
| paloma-redpajama | 9 | +0.03 [-0.12, +0.15] | +0.53 [+0.48, +0.59] | -0.48 [-0.74, -0.27] | +0.23 | +0.47 [+0.20, +0.75] | 0.80 | smooth, shape unclear |

## D2: 6 Paloma models (replication) — secondary: fixed layer 8

**Verdict: PARTLY RIGHT (mixed).** Classes: {"can't tell": 1, 'length staircase': 1, 'mixed': 3, 'smooth, shape unclear': 1}. Median LS = +0.32 (LS at 100: +0.48, at 1000: +0.24); median λ = +0.04.

Tokenizer sub-test: input-format pattern (no step at 100, step at 1000) in 0/6; learned-length pattern (step at both) in 1/6.

| model | layer | LS | LS at 100 | LS at 1000 | J | λ inside | ρ | class |
|---|---|---|---|---|---|---|---|---|
| paloma-c4 | 8 | -0.49 [-0.89, -0.26] | -1.40 [-2.15, -0.95] | +0.41 [+0.32, +0.48] | +0.17 | +1.80 [+1.55, +1.98] | 0.70 | can't tell |
| paloma-dolma | 8 | +0.37 [+0.33, +0.41] | +0.51 [+0.47, +0.54] | +0.23 [+0.18, +0.28] | +0.24 | -0.23 [-0.40, -0.03] | 0.94 | mixed |
| paloma-falcon-refinedweb | 8 | +0.34 [+0.27, +0.41] | +0.61 [+0.58, +0.65] | +0.07 [-0.05, +0.17] | +0.64 | -0.50 [-0.70, -0.33] | 0.89 | mixed |
| paloma-mc4 | 8 | +0.54 [+0.51, +0.56] | +0.52 [+0.50, +0.55] | +0.55 [+0.52, +0.57] | +0.50 | +0.25 [+0.12, +0.38] | 0.96 | length staircase |
| paloma-pile | 8 | +0.30 [+0.26, +0.32] | +0.35 [+0.32, +0.38] | +0.24 [+0.20, +0.27] | +0.18 | -0.17 [-0.28, -0.03] | 0.96 | mixed |
| paloma-redpajama | 8 | +0.15 [+0.08, +0.21] | +0.45 [+0.41, +0.50] | -0.16 [-0.27, -0.06] | +0.21 | +0.32 [+0.15, +0.51] | 0.89 | smooth, shape unclear |

### E06_Paloma: all layers (median over models, point estimates, exploratory)

| layer | LS | LS 100 | LS 1000 | J | λ | ρ |
|---|---|---|---|---|---|---|
| 1 | -0.10 | -0.40 | +0.76 | -2.27 | +0.88 | 0.03 |
| 2 | +0.36 | +0.34 | +0.38 | +0.21 | +0.68 | 0.44 |
| 3 | +0.32 | +0.44 | +0.40 | +0.37 | +0.15 | 0.79 |
| 4 | +0.15 | +0.33 | +0.23 | +0.39 | -0.12 | 0.94 |
| 5 | +0.20 | +0.41 | -0.03 | +0.29 | -0.21 | 0.90 |
| 6 | +0.23 | +0.43 | +0.13 | +0.36 | -0.23 | 0.95 |
| 7 | +0.34 | +0.41 | +0.31 | +0.26 | +0.09 | 0.95 |
| 8 | +0.32 | +0.48 | +0.24 | +0.22 | +0.04 | 0.91 |
| 9 | +0.26 | +0.46 | +0.22 | +0.24 | -0.03 | 0.92 |
| 10 | +0.24 | +0.37 | +0.16 | +0.25 | -0.05 | 0.95 |
| 11 | +0.10 | +0.22 | +0.01 | +0.20 | -0.07 | 0.92 |
| 12 | +0.08 | +0.16 | +0.03 | +0.02 | -0.02 | 0.91 |
| 13 | -0.52 | -0.30 | -0.74 | -0.21 | -0.17 | 0.81 |
| 14 | -3.34 | -2.71 | -52.98 | -5.13 | -0.72 | 0.21 |
| 15 | -2.81 | -2.67 | +3.92 | -9.36 | -0.97 | 0.14 |
| 16 | -10.13 | -0.55 | -19.71 | -12.29 | -1.00 | 0.15 |

## D3: Pythia and Qwen, every number 10–9999 (replication + tokenizer contrast)

| model | reading | layer | LS | LS at 100 | LS at 1000 | J | λ inside | ρ | class | R 3-digit | R 4-digit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-1.4b | copy '=' (C1, beta-like) | 9 | +0.23 [+0.16, +0.31] | +0.40 [+0.34, +0.45] | +0.07 [-0.04, +0.18] | +0.18 | +0.05 [-0.25, +0.35] | 0.97 | mixed | +10.4 | +22.1 |
| pythia-1.4b | sentence (AVG5, number token) | 18 | +0.31 [+0.25, +0.35] | +0.44 [+0.40, +0.48] | +0.18 [+0.09, +0.23] | +0.13 | +0.25 [+0.00, +0.45] | 0.97 | mixed | +17.5 | -12.9 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 4 | +0.69 [+0.60, +0.75] | +0.85 [+0.80, +0.88] | +0.53 [+0.40, +0.62] | +0.69 | +0.50 [-0.30, +1.60] | 0.94 | length staircase | +4.7 | -1.4 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 28 | +0.18 [+0.11, +0.24] | +0.41 [+0.37, +0.46] | -0.04 [-0.16, +0.04] | +0.11 | -0.40 [-0.60, -0.25] | 0.97 | smooth, log inside | -45.6 | -36.9 |

### D3: all layers (point estimates, exploratory)

| model | reading | layer | LS | LS 100 | LS 1000 | λ | R3 | ρ |
|---|---|---|---|---|---|---|---|---|
| pythia-1.4b | copy '=' (C1, beta-like) | 3 | +0.26 | +0.55 | -0.03 | -0.38 | +7.2 | 0.92 |
| pythia-1.4b | copy '=' (C1, beta-like) | 6 | +0.34 | +0.46 | +0.21 | +0.03 | +15.9 | 0.97 |
| pythia-1.4b | copy '=' (C1, beta-like) | 9 | +0.23 | +0.40 | +0.07 | +0.05 | +10.4 | 0.97 |
| pythia-1.4b | copy '=' (C1, beta-like) | 12 | -0.03 | +0.30 | -0.37 | +0.05 | +6.3 | 0.95 |
| pythia-1.4b | copy '=' (C1, beta-like) | 15 | +0.31 | +0.25 | +0.36 | -0.15 | -2.5 | 0.90 |
| pythia-1.4b | copy '=' (C1, beta-like) | 18 | +0.41 | +0.29 | +0.51 | +0.65 | +0.8 | 0.87 |
| pythia-1.4b | copy '=' (C1, beta-like) | 20 | +0.33 | +0.32 | +0.36 | +0.65 | +0.3 | 0.88 |
| pythia-1.4b | copy '=' (C1, beta-like) | 24 | +0.09 | +0.23 | -0.04 | +1.00 | +0.7 | 0.86 |
| pythia-1.4b | sentence (AVG5, number token) | 1 | +0.71 | +0.96 | +0.58 | +2.00 | -9.6 | 0.28 |
| pythia-1.4b | sentence (AVG5, number token) | 2 | +0.06 | -0.05 | +0.16 | +2.00 | +0.1 | 0.74 |
| pythia-1.4b | sentence (AVG5, number token) | 3 | +0.18 | +0.15 | +0.20 | +1.50 | +6.4 | 0.84 |
| pythia-1.4b | sentence (AVG5, number token) | 4 | +0.22 | +0.18 | +0.25 | +1.17 | +5.8 | 0.94 |
| pythia-1.4b | sentence (AVG5, number token) | 5 | +0.25 | +0.20 | +0.29 | +0.85 | +4.5 | 0.96 |
| pythia-1.4b | sentence (AVG5, number token) | 6 | +0.23 | +0.14 | +0.33 | +0.68 | +3.1 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 7 | +0.26 | +0.14 | +0.37 | +0.60 | +3.4 | 0.96 |
| pythia-1.4b | sentence (AVG5, number token) | 8 | +0.27 | +0.14 | +0.40 | +0.55 | +2.8 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 9 | +0.29 | +0.18 | +0.40 | +0.55 | +3.1 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 10 | +0.33 | +0.21 | +0.43 | +0.62 | +3.4 | 0.96 |
| pythia-1.4b | sentence (AVG5, number token) | 11 | +0.32 | +0.23 | +0.40 | +0.62 | +3.9 | 0.96 |
| pythia-1.4b | sentence (AVG5, number token) | 12 | +0.34 | +0.26 | +0.42 | +0.55 | +3.7 | 0.96 |
| pythia-1.4b | sentence (AVG5, number token) | 13 | +0.34 | +0.30 | +0.36 | +0.55 | +4.5 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 14 | +0.30 | +0.32 | +0.29 | +0.42 | +5.0 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 15 | +0.30 | +0.37 | +0.23 | +0.35 | +6.4 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 16 | +0.29 | +0.40 | +0.19 | +0.28 | +7.1 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 17 | +0.30 | +0.43 | +0.16 | +0.25 | +10.9 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 18 | +0.31 | +0.44 | +0.18 | +0.25 | +17.5 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 19 | +0.30 | +0.44 | +0.16 | +0.25 | +17.8 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 20 | +0.31 | +0.44 | +0.17 | +0.20 | +16.5 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 21 | +0.32 | +0.45 | +0.20 | +0.20 | +24.5 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 22 | +0.35 | +0.46 | +0.23 | +0.17 | +17.1 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 23 | +0.37 | +0.47 | +0.27 | +0.15 | +11.3 | 0.97 |
| pythia-1.4b | sentence (AVG5, number token) | 24 | +0.38 | +0.47 | +0.29 | +0.10 | +7.0 | 0.97 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 4 | +0.69 | +0.85 | +0.53 | +0.50 | +4.7 | 0.94 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 7 | +0.36 | +0.64 | +0.07 | +0.45 | -44.1 | 0.97 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 11 | +0.38 | +0.51 | +0.25 | +0.50 | +3.5 | 0.96 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 14 | +0.45 | +0.49 | +0.41 | +0.00 | +4.6 | 0.95 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 18 | +0.51 | +0.46 | +0.57 | +0.05 | -10.6 | 0.95 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 21 | +0.39 | +0.36 | +0.41 | -0.05 | -16.5 | 0.97 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 24 | -8.18 | -7.97 | -9.41 | -0.62 | -0.6 | 0.26 |
| qwen2.5-1.5b | copy '=' (C1, beta-like) | 28 | +0.47 | +0.52 | -1.74 | +2.00 | +0.7 | 0.08 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 1 | -2.34 | +0.91 | -5.59 | +2.00 | +28.6 | 0.72 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 2 | +0.48 | +0.75 | +0.20 | +0.35 | +18.5 | 0.95 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 3 | +0.60 | +0.76 | +0.43 | +0.25 | -15.3 | 0.91 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 4 | +0.61 | +0.69 | +0.53 | +0.30 | +13.4 | 0.94 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 5 | +0.55 | +0.51 | +0.57 | -0.35 | +6.1 | 0.94 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 6 | +0.48 | +0.57 | +0.38 | +0.05 | +13.7 | 0.95 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 7 | +0.41 | +0.53 | +0.30 | +0.35 | +23.1 | 0.97 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 8 | +0.32 | +0.51 | +0.13 | +0.40 | +19.6 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 9 | +0.27 | +0.48 | +0.07 | +0.40 | +11.9 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 10 | +0.27 | +0.47 | +0.07 | +0.45 | +9.7 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 11 | +0.29 | +0.45 | +0.12 | +0.45 | +11.4 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 12 | +0.33 | +0.48 | +0.17 | +0.55 | +7.1 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 13 | +0.31 | +0.48 | +0.12 | +0.45 | +7.0 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 14 | +0.35 | +0.44 | +0.27 | +0.28 | +6.6 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 15 | +0.34 | +0.44 | +0.24 | +0.35 | +5.4 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 16 | +0.34 | +0.45 | +0.24 | +0.30 | +6.4 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 17 | +0.36 | +0.46 | +0.27 | +0.30 | +7.2 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 18 | +0.38 | +0.46 | +0.28 | +0.30 | +7.3 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 19 | +0.30 | +0.47 | +0.14 | +0.15 | +12.2 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 20 | +0.37 | +0.46 | +0.28 | +0.35 | +7.4 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 21 | +0.39 | +0.46 | +0.32 | +0.35 | +7.6 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 22 | +0.34 | +0.50 | +0.19 | +0.20 | +16.0 | 0.97 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 23 | +0.37 | +0.53 | +0.20 | +0.10 | +66.0 | 0.97 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 24 | +0.34 | +0.52 | +0.17 | -0.05 | -99.7 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 25 | +0.33 | +0.51 | +0.15 | -0.15 | -36.7 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 26 | +0.26 | +0.48 | +0.04 | -0.10 | -35.7 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 27 | +0.28 | +0.51 | +0.06 | -0.10 | +64.9 | 0.98 |
| qwen2.5-1.5b | sentence (AVG5, number token) | 28 | +0.18 | +0.41 | -0.04 | -0.40 | -45.6 | 0.97 |

## Power (synthetic plants)

| design | noise | plant | correct | LS | λ | classes |
|---|---|---|---|---|---|---|
| E06 (bands, 3 seeds) | 0.1 | professor toy (one-hot only) | 1.00 | +0.99 | +0.67 | {"length staircase": 20} |
| E06 (bands, 3 seeds) | 0.1 | length staircase | 1.00 | +0.99 | +0.10 | {"length staircase": 20} |
| E06 (bands, 3 seeds) | 0.1 | smooth log | 1.00 | +0.03 | -0.01 | {"smooth, log inside": 20} |
| E06 (bands, 3 seeds) | 0.1 | smooth, linear inside | 1.00 | +0.10 | +0.99 | {"smooth, linear inside": 20} |
| E06 (bands, 3 seeds) | 0.1 | mixed (37.5% length) | 1.00 | +0.39 | +0.03 | {"mixed": 20} |
| E06 (bands, 3 seeds) | 0.3 | professor toy (one-hot only) | 1.00 | +0.99 | +0.02 | {"length staircase": 20} |
| E06 (bands, 3 seeds) | 0.3 | length staircase | 1.00 | +1.00 | +0.28 | {"length staircase": 20} |
| E06 (bands, 3 seeds) | 0.3 | smooth log | 1.00 | +0.04 | -0.01 | {"smooth, log inside": 20} |
| E06 (bands, 3 seeds) | 0.3 | smooth, linear inside | 1.00 | +0.11 | +0.98 | {"smooth, linear inside": 20} |
| E06 (bands, 3 seeds) | 0.3 | mixed (37.5% length) | 1.00 | +0.40 | -0.03 | {"mixed": 20} |
| E06 (bands, 3 seeds) | 0.6 | professor toy (one-hot only) | 1.00 | +1.04 | +0.11 | {"length staircase": 20} |
| E06 (bands, 3 seeds) | 0.6 | length staircase | 1.00 | +1.00 | +0.10 | {"length staircase": 20} |
| E06 (bands, 3 seeds) | 0.6 | smooth log | 1.00 | +0.04 | +0.03 | {"smooth, log inside": 20} |
| E06 (bands, 3 seeds) | 0.6 | smooth, linear inside | 1.00 | +0.09 | +1.01 | {"smooth, linear inside": 20} |
| E06 (bands, 3 seeds) | 0.6 | mixed (37.5% length) | 1.00 | +0.41 | +0.01 | {"mixed": 20} |
| E13 (every number 10-9999) | 0.1 | professor toy (one-hot only) | 1.00 | +1.01 | +0.65 | {"length staircase": 20} |
| E13 (every number 10-9999) | 0.1 | length staircase | 1.00 | +0.99 | -0.95 | {"length staircase": 20} |
| E13 (every number 10-9999) | 0.1 | smooth log | 1.00 | +0.03 | +0.00 | {"smooth, log inside": 20} |
| E13 (every number 10-9999) | 0.1 | smooth, linear inside | 1.00 | -0.02 | +1.00 | {"smooth, linear inside": 20} |
| E13 (every number 10-9999) | 0.1 | mixed (37.5% length) | 1.00 | +0.40 | -0.10 | {"mixed": 20} |
| E13 (every number 10-9999) | 0.3 | professor toy (one-hot only) | 0.70 | +1.02 | +0.93 | {"length staircase": 14, "can't tell": 6} |
| E13 (every number 10-9999) | 0.3 | length staircase | 1.00 | +1.00 | -0.10 | {"length staircase": 20} |
| E13 (every number 10-9999) | 0.3 | smooth log | 1.00 | +0.05 | -0.07 | {"smooth, log inside": 20} |
| E13 (every number 10-9999) | 0.3 | smooth, linear inside | 1.00 | -0.03 | +1.00 | {"smooth, linear inside": 20} |
| E13 (every number 10-9999) | 0.3 | mixed (37.5% length) | 1.00 | +0.41 | +0.00 | {"mixed": 20} |
| E13 (every number 10-9999) | 0.6 | professor toy (one-hot only) | 0.00 | +0.99 | +1.27 | {"can't tell": 10, "no number line": 10} |
| E13 (every number 10-9999) | 0.6 | length staircase | 1.00 | +1.01 | +1.40 | {"length staircase": 20} |
| E13 (every number 10-9999) | 0.6 | smooth log | 1.00 | +0.04 | +0.05 | {"smooth, log inside": 20} |
| E13 (every number 10-9999) | 0.6 | smooth, linear inside | 1.00 | +0.00 | +1.02 | {"smooth, linear inside": 20} |
| E13 (every number 10-9999) | 0.6 | mixed (37.5% length) | 1.00 | +0.39 | +0.00 | {"mixed": 20} |
