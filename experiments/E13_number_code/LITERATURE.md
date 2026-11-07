# E13 literature check (2026-11-07, before Stage 1)

Read: the arXiv abstracts of all papers below and the full HTML text of the first six. Searches covered:
- per-digit geometry;
- carry/successor geometry;
- ultrametric or hyperbolic number embeddings;
- lexical vs compositional number codes;
- subspace dissociation.

This is not a systematic review. Anything marked "not found" means not found in these searches.

| Paper | What it did | Overlap with E13 |
|---|---|---|
| Levy & Geva 2024/25, arXiv 2410.11781, *LMs Encode Numbers Using Digit Representations in Base 10* | Llama-3-8B and Mistral-7B, numbers 0–999. Supervised **circular probes** per digit (also linear probes, other bases). Causal +5 mod 10 edits on digit representations change outputs. | Closest to H1. They assume and probe a circle at every position. They do **not** compare line / circle / simplex per position, use no unsupervised geometry, and do not test whether the code changes with position. Note: a supervised probe can map *any* categorical (simplex) code onto a circle, so probe success is not evidence of circular geometry. |
| Kantamneni & Tegmark 2025, arXiv 2502.00873, *LMs Use Trigonometry to Do Addition* | GPT-J and others, single-token integers 0–360. Generalised helix: linear + periods 2, 5, 10, 100. "Clock" algorithm for a + b, with causal checks. | Rival R-c (circle/helix). Carries appear only as a hypothesis for errors. Single-token range only. |
| Fu, Zhou, Belkin, Sharan, Jia 2026, arXiv 2604.20817, *Convergent Evolution* | Many architectures learn Fourier features at periods 2, 5, 10. Only some are **geometrically separable** (mod-T linearly classifiable). Tokenizer, data and optimizer matter. | Related to R-c and to the tokenizer effect. Not about position-resolved codes or carries. |
| Karkada et al. 2026, arXiv 2602.15029, *Symmetry in language statistics shapes the geometry of model representations* | Translation symmetry in co-occurrence statistics explains month circles and year lines. | Theory for why a line or circle forms. Not digit positions. |
| Bassi & Tomar 2026, arXiv 2607.04167, *Geometry of Ordinal Representations in LMs* | Gemma-2-2B/9B, Qwen3-4B. 1-D manifolds with place-cell SAE tiling appear for locally computable ordinals. Numeric magnitude gives higher-dimensional or incoherent representations. | Consistent with E12 (no clean 1-D line). Not per digit. |
| Cacioli 2026, arXiv 2603.28258, *Categorical Perception in LLM Hidden States* | RSA in 6 models: log distance + a "boundary boost" at the digit-count transitions 10 and 100 fits better than continuous distance. | Relevant to the **exponent code** (digit count). Not per-position geometry, not carries. |
| Wen et al. 2026, arXiv 2606.03645, *The Shape of Addition* | Qwen3-4B, 3-operand 10-digit addition: states anchored by digits plus continuous "carry fibers"; errors are threshold crossings. | Carries **during an addition computation**. E13-H2 asks about carries in the geometry of *reading* n vs n + 1, which was not found. |
| AlquBoj et al. 2025, arXiv 2502.16147 | PCA/PLS: sublinear (log-like) spacing. | Contrasts with E12 (near-linear once digit length is fixed). |
| Hu, Niu, Varma 2026, arXiv 2602.06843, *The Representational Geometry of Number* | Task-specific subspaces; magnitude and parity on separable linear directions, linearly mappable across tasks. | Relevant to H5 (dissociation). |
| arXiv 2512.20926 (hyperbolicity/ultrametricity of LLM embeddings, general text) | δ-hyperbolicity and ultrametricity of text embeddings. | Method only. **No number-specific tree / ultrametric study found.** |
| Informal blog (bhasa-chat.netlify.app, "LLMs counting digits, not storing a number line") | Qwen2.5-1.5B: erasing a "magnitude" subspace drops comparison to chance; leading-digit subspace partly distinct. | Unreviewed. Closest to H5. Must be cited or checked if H5 is run. |

**Not found:**
1. A position-by-position comparison of line / circle / simplex digit codes with planted-power controls.
2. Carry spikes or a successor operator in the representation of n.
3. Tree (prefix / ultrametric) vs line tests for numbers.
4. A lexical vs compositional decomposition of number vectors.

Novelty claims must still name Levy & Geva (per-digit codes exist) and Kantamneni & Tegmark / Fu et al.
(periodic features).
