"""Novelty check (brief section 12): papers citing the four anchor papers, newest first, with
those touching topology / persistent homology of truth representations, or non-linear tracking
of truth across pretraining, flagged. Needs internet access to api.semanticscholar.org.

    python -m tt.novelty      # -> results/related_work.md
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

from . import config as C

ANCHORS = {
    "Geometry of Truth (Marks & Tegmark)": "arXiv:2310.06824",
    "Truth is Universal (Buerger et al.)": "arXiv:2407.12831",
    "Emergence of Linear Truth Encodings (Ravfogel et al.)": "arXiv:2510.15804",
    "Shape of Adversarial Influence (Fay et al.)": "arXiv:2505.20435",
}
TOPO = re.compile(r"topolog|persistent homolog|persistence diagram|barcode|betti|vietoris|manifold", re.I)
TRUTH = re.compile(r"truth|factual|hallucinat|honest|lie|deception|knowledge", re.I)
DYN = re.compile(r"pre-?train\w* (checkpoint|dynamic)|checkpoints|training dynamics|across (pre)?training", re.I)
API = "https://api.semanticscholar.org/graph/v1/paper/{}/citations?fields=title,year,publicationDate,abstract,externalIds&limit=1000"


def get(url):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # 429 rate limits are common without a key
            time.sleep(2 ** (a + 2))
    raise RuntimeError(url)


def main():
    lines = ["# Related work found (Semantic Scholar citations, newest first)\n",
             "Flags: **T** topology/PH terms, **F** truth/factuality terms, **D** training-dynamics terms. "
             "T+F or F+D (non-linear) papers need a manual read.\n"]
    for name, pid in ANCHORS.items():
        data = get(API.format(pid)).get("data", [])
        papers = [d["citingPaper"] for d in data if d.get("citingPaper", {}).get("title")]
        papers.sort(key=lambda p: (p.get("publicationDate") or str(p.get("year") or "")), reverse=True)
        lines.append(f"\n## Citing {name} ({len(papers)})\n")
        for p in papers:
            text = (p.get("title") or "") + " " + (p.get("abstract") or "")
            flags = "".join(f for f, rx in (("T", TOPO), ("F", TRUTH), ("D", DYN)) if rx.search(text))
            star = " ⚑" if ("T" in flags and "F" in flags) or ("F" in flags and "D" in flags) else ""
            arx = (p.get("externalIds") or {}).get("ArXiv")
            lines.append(f"- {p.get('publicationDate') or p.get('year')} [{flags or '-'}]{star} {p['title']}"
                         + (f" (arXiv {arx})" if arx else ""))
        time.sleep(3)
    (C.RESULTS / "related_work.md").write_text("\n".join(lines) + "\n")
    print("wrote", C.RESULTS / "related_work.md")


if __name__ == "__main__":
    main()
