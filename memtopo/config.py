"""Config loading, output paths and logging for memtopo."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "memtopo.yaml"


def load_config(path: str | os.PathLike | None = None, out_root: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if out_root:
        cfg["out_root"] = out_root
    os.environ.setdefault("HF_HOME", cfg.get("hf_home", "/workspace/hf"))
    return cfg


def model_spec(cfg: dict, key: str) -> dict:
    spec = dict(cfg["models"][key])
    for bad in cfg["forbidden_models"]:
        if bad in spec["name"]:
            raise ValueError(f"{spec['name']} is excluded by the brief ({bad})")
    spec["key"] = key
    return spec


def layers_for(n_layers: int, ref16: list[int]) -> list[int]:
    """Same relative depths for every model: round-half-up(L * l / 16), 1-indexed."""
    out = [int(n_layers * l / 16 + 0.5) for l in ref16]
    if len(set(out)) != len(out) or out[-1] != n_layers or out[0] < 1:
        raise ValueError(f"bad layer list {out} for {n_layers} layers")
    return out


@dataclass
class Paths:
    root: Path

    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def shards(self) -> Path: return self.root / "data" / "shards"
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def figures(self) -> Path: return self.root / "figures"

    def scans(self, model: str, step: int) -> Path:
        return self.root / "scans" / model / f"step{step}"

    def features(self, model: str, sf: int) -> Path:
        return self.root / "features" / model / f"S{sf}"

    def results(self, model: str, sf: int) -> Path:
        return self.root / "results" / model / f"S{sf}"

    @property
    def pool(self) -> Path: return self.data / "pool.parquet"
    @property
    def gates(self) -> Path: return self.root / "gates.json"

    def make(self) -> "Paths":
        for p in (self.data, self.shards, self.logs, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def paths(cfg: dict) -> Paths:
    return Paths(Path(cfg["out_root"])).make()


def setup_logging(p: Paths, stage: str) -> logging.Logger:
    log = logging.getLogger("memtopo")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(), logging.FileHandler(p.logs / f"{stage}.log")):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def record_gate(p: Paths, name: str, value: dict) -> None:
    """Gates and checked facts (epoch boundary, alignment, prevalence) go to gates.json."""
    g = json.loads(p.gates.read_text()) if p.gates.exists() else {}
    g[name] = value
    p.gates.write_text(json.dumps(g, indent=2, default=str))


def read_gates(p: Paths) -> dict:
    return json.loads(p.gates.read_text()) if p.gates.exists() else {}
