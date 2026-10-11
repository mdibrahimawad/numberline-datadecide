"""Section 12 novelty check: papers citing the three anchor papers (Semantic Scholar Graph API),
newest first, flagging titles/abstracts that mention hidden-state geometry/topology or checkpoints."""

from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests

ANCHORS = {"biderman2023": "arXiv:2304.11158", "latent_memories": "arXiv:2406.14549",
           "fay2026": "arXiv:2505.20435"}
FLAG = re.compile(r"hidden[- ]state|representation|embedding|geometr|topolog|persistent homology|"
                  r"checkpoint|training dynamics|forecast|predict", re.I)
API = "https://api.semanticscholar.org/graph/v1/paper/{}/citations"


def citations(paper_id: str) -> list[dict]:
    out, offset = [], 0
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    while True:
        r = requests.get(API.format(paper_id), headers=headers, timeout=60,
                         params={"fields": "title,abstract,year,publicationDate,externalIds,url",
                                 "offset": offset, "limit": 1000})
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        data = r.json().get("data", [])
        out += [d["citingPaper"] for d in data]
        if len(data) < 1000:
            return out
        offset += 1000


def run(out_csv: str) -> pd.DataFrame:
    rows = []
    for anchor, pid in ANCHORS.items():
        for c in citations(pid):
            text = f"{c.get('title') or ''} {c.get('abstract') or ''}"
            hits = sorted({m.group(0).lower() for m in FLAG.finditer(text)})
            rows.append({"anchor": anchor, "title": c.get("title"), "date": c.get("publicationDate") or c.get("year"),
                         "url": c.get("url"), "flag_terms": ", ".join(hits),
                         "flag": bool(re.search(r"geometr|topolog|homology|hidden[- ]state|checkpoint", text, re.I))})
    df = pd.DataFrame(rows).sort_values("date", ascending=False, key=lambda s: s.astype(str))
    df.to_csv(out_csv, index=False)
    return df
