"""CPU smoke test on the tiny DataDecide model (needs huggingface.co access).

allenai/DataDecide-dolma1_7-60M, groups 1-4, k=5, one selection seed (42),
one eval seed (45), correct-output filter on. Every group must end with 5
accepted prompts or be flagged failed, and every prompt must end on '='.

    pip install "transformers==4.49.0" "ai2-olmo==0.6.0"
    python tests/test_datadecide_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datadecide_sweep import (
    recipe_slug,
    run_datadecide_model,
    verify_revisions,
    write_summary_csv,
)

MODEL = "allenai/DataDecide-dolma1_7-60M"
K = 5
OUT_DIR = Path("results/datadecide_smoke")


def test_datadecide_60m_smoke():
    problems = verify_revisions([MODEL], None)
    assert not problems, problems

    payload = run_datadecide_model(
        MODEL, None, groups=(1, 2, 3, 4), k=K,
        selection_seeds=(42,), eval_seeds=(45,), device="cpu", dtype="float32",
    )

    diag = payload["tokenization_diagnostics"]
    print(f"[smoke] last-token sample: {diag['samples'][0]['last_token_text']!r}; "
          f"{diag['prompts_ending_equals_token']}/{diag['prompt_count']} prompts end on '='")
    assert payload["final_token_is_equals"], diag["samples"][:3]

    per_seed = payload["selection"]["filter_stats"] + [
        {"seed": r["seed"], "groups": r["filter_stats"]} for r in payload["evaluation"]["runs"]
    ]
    for entry in per_seed:
        assert sorted(entry["groups"], key=int) == ["1", "2", "3", "4"]
        for group, st in entry["groups"].items():
            print(f"[smoke] seed {entry['seed']} group {group}: accepted {st['accepted']}/{K} "
                  f"tried {st['candidates_tried']} failed={st['failed']}")
            assert st["accepted"] == K or st["failed"], (entry["seed"], group, st)

    payload["recipe"] = recipe_slug(MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{payload['recipe']}.json").write_text(json.dumps(payload, indent=2))
    print(f"[smoke] selected layer {payload['selection']['selected_layer']}; "
          f"wrote {write_summary_csv(OUT_DIR)}")


if __name__ == "__main__":
    test_datadecide_60m_smoke()
    print("ok  test_datadecide_60m_smoke")
