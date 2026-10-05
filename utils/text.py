from __future__ import annotations

import re
from collections import Counter

DIGIT_TOKEN_PATTERN: re.Pattern[str] = re.compile(r"\b\d+\b")


def count_digit_tokens(text: str) -> Counter[str]:
    return Counter(DIGIT_TOKEN_PATTERN.findall(text))
