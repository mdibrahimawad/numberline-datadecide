"""Command line for E10. Run from the repo root on the pod, in the order of the brief's section 10:

  python -m memtopo.cli refs                       # clone/inspect reference repos (setup step)
  python -m memtopo.cli data                       # shards, epoch boundary, pool, B4  (gate 2)
  python -m memtopo.cli align --model 1b           # 200/200 token check               (gate 2)
  python -m memtopo.cli validate                   # topology code on synthetic clouds  (gate 4)
  python -m memtopo.cli labels --model 1b          # mem_T, mem_Sf, mem_final, fragility (gate 3)
  python -m memtopo.cli pilot --model 1b --sf 10000   # scans + features + analysis + figures (gate 5)
  python -m memtopo.cli report

Every stage skips outputs that already exist, so rerunning after an interruption resumes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from memtopo.config import load_config, paths, record_gate, setup_logging

REF_REPOS = {
    "pythia": "https://github.com/EleutherAI/pythia",
    "tracking-representation-dynamics-with-persistent-homology":
        "https://github.com/malhotranaman/tracking-representation-dynamics-with-persistent-homology",
    "memorisation-profiles": "https://github.com/pietrolesci/memorisation-profiles",
    "mimir": "https://github.com/iamgroot42/mimir",
}


def cmd_refs(cfg, args):
    """Clone the reference repos and record which exist and what was found in them."""
    from pathlib import Path

    refs = Path(cfg["refs_dir"])
    refs.mkdir(parents=True, exist_ok=True)
    found = {}
    for name, url in REF_REPOS.items():
        d = refs / name
        if not d.exists():
            r = subprocess.run(["git", "clone", "--depth", "1", url, str(d)], capture_output=True, text=True)
            if r.returncode:
                found[name] = f"clone failed: {r.stderr.strip()[-200:]}"
                continue
        found[name] = "ok"
    summ = refs / "tracking-representation-dynamics-with-persistent-homology" / "src" / "phlatent" / "topology" / "summary.py"
    if summ.exists():
        sys.path.insert(0, str(summ.parents[2]))
        try:
            from phlatent.topology.summary import FEATURE_NAMES as theirs  # type: ignore
            from memtopo.topology import FEATURE_NAMES as ours
            found["malhotra_feature_names"] = list(theirs)
            found["malhotra_n_features"] = len(theirs)
            found["ours_n_features"] = len(ours)
        except Exception as e:  # noqa: BLE001 - report whatever differs
            found["malhotra_summary_import"] = repr(e)
    else:
        found["malhotra_summary_py"] = "not at src/phlatent/topology/summary.py"
    record_gate(paths(cfg), "reference_repos", found)
    print(found)


def cmd_validate(cfg, args):
    """Gate 4: (a) noisy circle -> one long H1 bar; (b) Gaussian -> only short bars;
    (c) x10 scaling leaves normalised features unchanged; (d) W(diagram, itself) = 0."""
    import numpy as np

    from memtopo import topology as topo

    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 64)
    circle = np.c_[np.cos(th), np.sin(th), np.zeros((64, 30))] + 0.05 * rng.standard_normal((64, 32))
    dc = topo.diagrams(topo.normalise(circle))
    pers = np.sort(dc[1][:, 1] - dc[1][:, 0])[::-1]
    a = bool(len(pers) >= 1 and pers[0] > 0.5 and (len(pers) == 1 or pers[0] > 4 * pers[1]))
    dg = topo.diagrams(topo.normalise(rng.standard_normal((64, 2048))))
    gp = dg[1][:, 1] - dg[1][:, 0]
    b = bool(gp.size == 0 or gp.max() < 0.5 * pers[0])
    x = rng.standard_normal((64, 512))
    f1 = topo.summary(topo.diagrams(topo.normalise(x)))
    f2 = topo.summary(topo.diagrams(topo.normalise(10 * x + 3)))
    c = bool(np.allclose(f1, f2, rtol=1e-6, atol=1e-8))
    d = bool(topo.wasserstein(dc[1], dc[1]) == 0 and topo.wasserstein(dc[0], dc[0]) == 0)
    res = {"a_circle_one_long_bar": a, "circle_top2_pers": pers[:2].tolist(),
           "b_gaussian_short_bars": b, "gaussian_max_h1_pers": float(gp.max()) if gp.size else 0.0,
           "c_scale_invariant": c, "d_self_distance_zero": d, "pass": a and b and c and d}
    record_gate(paths(cfg), "topology_validation", res)
    print(res)
    if not res["pass"]:
        raise SystemExit("topology validation FAILED")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="memtopo")
    ap.add_argument("stage", choices=["refs", "data", "align", "validate", "labels", "casecontrol", "scans",
                                      "features", "analyze", "figures", "pilot", "report", "novelty"])
    ap.add_argument("--config")
    ap.add_argument("--out-root")
    ap.add_argument("--model", default="1b")
    ap.add_argument("--sf", type=int, default=10000)
    ap.add_argument("--target", default="mem_T", choices=["mem_T", "mem_final"])
    ap.add_argument("--skip-epoch", action="store_true", help="data: skip the epoch-boundary search")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.out_root)
    p = paths(cfg)
    log = setup_logging(p, args.stage)
    log.info("stage %s %s", args.stage, vars(args))

    from memtopo import analysis, figures, pipeline, report

    s = args.stage
    if s == "refs":
        cmd_refs(cfg, args)
    elif s == "validate":
        cmd_validate(cfg, args)
    elif s == "data":
        pipeline.stage_data(cfg, skip_epoch=args.skip_epoch)
    elif s == "align":
        pipeline.stage_align(cfg, args.model)
    elif s == "labels":
        pipeline.stage_labels(cfg, args.model)
    elif s == "casecontrol":
        pipeline.stage_casecontrol(cfg, args.model)
    elif s == "scans":
        pipeline.stage_scans(cfg, args.model, args.sf)
    elif s == "features":
        pipeline.stage_features(cfg, args.model, args.sf)
    elif s == "analyze":
        analysis.run_ladder(cfg, args.model, args.sf, args.target)
        analysis.residual_analysis(cfg, args.model, args.sf, args.target)
        analysis.effects(cfg, args.model, args.sf, args.target)
        if args.target == "mem_T":
            analysis.gaussian_floor(cfg, args.model, args.sf)
    elif s == "figures":
        figures.all_figures(cfg)
    elif s == "pilot":
        pipeline.stage_casecontrol(cfg, args.model)
        pipeline.stage_scans(cfg, args.model, args.sf)
        pipeline.stage_features(cfg, args.model, args.sf)
        main(["analyze", "--model", args.model, "--sf", str(args.sf)] + (["--config", args.config] if args.config else [])
             + (["--out-root", args.out_root] if args.out_root else []))
        figures.all_figures(cfg)
        print(report.write_summary(cfg))
    elif s == "report":
        print(report.write_summary(cfg))
    elif s == "novelty":
        from memtopo import novelty
        df = novelty.run(str(p.root / "results" / "related_work.csv"))
        print(df[df["flag"]].head(40).to_string())


if __name__ == "__main__":
    main()
