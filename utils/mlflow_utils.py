from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping

import mlflow


def setup_tracking(tracking_uri: str | None, experiment_name: str) -> None:
    if tracking_uri is None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{os.path.abspath('mlflow.db')}",
        )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(run_name: str | None, tags: Mapping[str, str] | None = None) -> Iterator[mlflow.ActiveRun]:
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(dict(tags))
        yield run


def log_params(params: Mapping[str, object]) -> None:
    mlflow.log_params(dict(params))


def log_metric(key: str, value: float, step: int | None = None) -> None:
    mlflow.log_metric(key, float(value), step=step)


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
