from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from utils.text import count_digit_tokens


@dataclass
class MultiProgressUpdate:
    docs_seen: int
    total_matches: int
    counts: Counter[str] = field(default_factory=Counter)


@dataclass
class MultiCountResult:
    docs_seen: int
    total_matches: int
    counts: Counter[str]


def count_all_numbers_in_stream(
    examples: Iterable[dict],
    max_docs: int,
    text_key: str = "text",
    log_every: int = 10_000,
    progress_callback: Callable[[MultiProgressUpdate], None] | None = None,
) -> Iterator[MultiProgressUpdate]:
    counts: Counter[str] = Counter()
    total_matches = 0
    docs_seen = 0

    for example in examples:
        text = example.get(text_key, "")
        if not isinstance(text, str):
            docs_seen += 1
            continue

        doc_counts = count_digit_tokens(text.lower())
        counts.update(doc_counts)
        total_matches += sum(doc_counts.values())
        docs_seen += 1

        if docs_seen % log_every == 0:
            update = MultiProgressUpdate(
                docs_seen=docs_seen,
                total_matches=total_matches,
                counts=Counter(counts),
            )
            if progress_callback is not None:
                progress_callback(update)
            yield update

        if docs_seen >= max_docs:
            break

    final = MultiProgressUpdate(
        docs_seen=docs_seen,
        total_matches=total_matches,
        counts=counts,
    )
    if progress_callback is not None:
        progress_callback(final)
    yield final


def run_multi_count(
    examples: Iterable[dict],
    max_docs: int,
    text_key: str = "text",
    log_every: int = 10_000,
    progress_callback: Callable[[MultiProgressUpdate], None] | None = None,
) -> MultiCountResult:
    last: MultiProgressUpdate | None = None
    for update in count_all_numbers_in_stream(
        examples=examples,
        max_docs=max_docs,
        text_key=text_key,
        log_every=log_every,
        progress_callback=progress_callback,
    ):
        last = update

    if last is None:
        return MultiCountResult(docs_seen=0, total_matches=0, counts=Counter())

    return MultiCountResult(
        docs_seen=last.docs_seen,
        total_matches=last.total_matches,
        counts=last.counts,
    )


def filter_to_range(counts: Counter[str], lo: int = 0, hi: int = 1000) -> dict[int, int]:
    # "1", "01", "001" all parse to int 1 — accumulate across these aliases so
    # small-integer counts don't collapse to whichever padded form appears last.
    out: dict[int, int] = {}
    for token, n in counts.items():
        try:
            k = int(token)
        except ValueError:
            continue
        if lo <= k <= hi:
            out[k] = out.get(k, 0) + n
    return out
