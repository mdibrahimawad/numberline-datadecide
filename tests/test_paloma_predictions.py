"""E09 predictions: the frozen lines applied to fake corpus summaries, checked by hand.

    python tests/test_paloma_predictions.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.paloma_predictions as pp  # noqa: E402


def test_predictions_follow_the_frozen_lines_and_flag_extrapolation():
    frozen = json.loads(pp.FROZEN.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "c4").mkdir()
        (root / "c4" / "summary.json").write_text(json.dumps(
            {"alpha_ols": -1.4614, "S_K4000": 5617.0, "R_K4000": 5184}))
        assert pp.main(["--counts", str(root)]) == 0
        row = json.loads((root / "predictions.json").read_text())["predictions"][0]
    a = frozen["lines"]["beta_cont"]["alpha_ols"]
    assert abs(row["pred_beta_cont_from_alpha_ols"] - (a["intercept"] + a["slope"] * -1.4614)) < 1e-12
    assert abs(row["pred_beta_cont_from_alpha_ols"] - 0.9283) < 1e-3     # by hand: 3.0752 - 1.4691 * 1.4614
    assert abs(row["pred_beta_cont_from_S"] - 0.9851) < 1e-3              # -0.1584 + 0.00020356 * 5617
    assert abs(row["pred_beta_cont_from_R"] - 1.2200) < 1e-3              # 2.5905 - 0.00026436 * 5184
    assert row["model"] == "allenai/paloma-1b-baseline-c4"
    # DataDecide: alpha_ols -1.762..-1.429, S 2970..5369, R 5707..7735
    assert row["alpha_ols_outside_datadecide"] is False and row["S_outside_datadecide"] is True
    assert row["R_outside_datadecide"] is True


if __name__ == "__main__":
    test_predictions_follow_the_frozen_lines_and_flag_extrapolation()
    print("ok")
