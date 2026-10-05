from __future__ import annotations

from typing import Iterator

from datasets import iterable_dataset as _iterable_dataset
from datasets import load_dataset


def _patch_datasets_torch_share_memory() -> None:
    if getattr(_iterable_dataset, "_maybe_share_with_torch_persistent_workers", None) is None:
        return
    _iterable_dataset._maybe_share_with_torch_persistent_workers = (  # type: ignore[attr-defined]
        lambda value: value
    )


_patch_datasets_torch_share_memory()


def load_pile_stream(
    dataset_name: str = "monology/pile-uncopyrighted",
    revision: str | None = None,
    data_dir: str | None = None,
    split: str = "train",
) -> Iterator[dict]:
    kwargs: dict = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    return load_dataset(dataset_name, **kwargs)
