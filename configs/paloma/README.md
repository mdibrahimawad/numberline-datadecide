# Paloma baseline training configs (E09)

Unchanged copies of the six OLMo training configs behind the Paloma 1B baselines, from
`github.com/allenai/OLMo` at commit `1f2f02052d2a5ecba82ff45bbfc731651b1e7d29` (the commit
the Paloma README names as the training code). `paloma_data_map.json` extracts what matters
for counting numbers:

| Model (`allenai/paloma-1b-baseline-…`) | Config | Training files | Data seed | Batch × length | Steps |
|---|---|---|---|---|---|
| c4 | `v1_5-mix-small-mitch-ish-c4-lumi.yaml` | 233 | 6198 | 2048 × 2048 | 35,000 |
| mc4 | `v1_5-mix-small-mitch-ish-mc4-lumi.yaml` | 3,136 | 6198 | 2048 × 2048 | 35,000 |
| redpajama | `v1_5-mix-small-mitch-ish-rpj-lumi.yaml` | 1,185 | 6198 | 2048 × 2048 | 35,000 |
| pile | `v1_5-mix-small-mitch-ish-pile-fixed-lumi.yaml` | 300 | 6198 | 2048 × 2048 | 35,000 |
| falcon-refinedweb | `v1_5-mix-small-mitch-ish-falcon-mcli.yaml` | 724 | 6198 | 2048 × 2048 | 739,328 in config ("150B" run name) |
| dolma | `v1_5-mix-small-mitch-ish-mcli.yaml` | 249 | 6198 | 2048 × 2048 | 739,328 in config ("150B" run name) |

35,000 steps × 2048 × 2048 tokens = 146.8B tokens. The falcon and dolma configs list a longer
duration but are named "150B"; the released checkpoint revisions show where they stopped.

Tokenizer `allenai_eleuther-ai-gpt-neox-20b-pii-special.json`, EOS = 0 (DataDecide uses 50279).
Training files are tokenized, decontaminated `.npy` files under `preprocessed/<corpus>/…`;
in the configs they sit on Ai2 storage (S3 / LUMI). Whether they are publicly downloadable
(e.g. from `olmo-data.org`, where OLMo's own Dolma v1.5 files are hosted) decides whether
E09 can count the exact training stream (same loader as DataDecide: per-file 2048-token
chunks, PCG64 shuffle with seed 6198) or must use a random sample.
