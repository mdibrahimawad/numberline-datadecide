"""Picture of the replicated shape-search result: class centroids of n mod 12 and n mod 60 (1000-9999), 2-D PCA view.
python -m src.e11_shape_figure -> results/e11_shape_search/fig_mod_clusters.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.e10_graded import layer_files
from src.e10_graded_analysis import INK, _style
from src.e11_shape_search import candidates, class_means, residual

DATA = "/home/user/mdibrahimawad/topology_number_line/data"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e", "#0d366b"]


def main():
    _style(plt)
    C = candidates()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    for r, (m, d, name) in enumerate((("crystal", 0, "LLM360 Crystal, ~25% depth"), ("olmo2-1b", 0, "OLMo-2, ~25% depth"))):
        X = np.load(layer_files(DATA, m)[d])["X"].astype(np.float32)
        for c, k in enumerate((12, 60)):
            _, pop, lab, grp = C[f"mod {k} (1000-9999)"]
            V = residual(X, pop, grp); cl = np.unique(lab); M = class_means(V, lab, cl)
            Z = M - M.mean(0); Z = Z @ np.linalg.svd(Z, full_matrices=False)[2][:2].T
            ax = axes[r, c]
            if k == 12:
                for par, col, lb in ((0, CAT[0], "even classes"), (1, CAT[1], "odd classes")):
                    i = cl % 2 == par; ax.scatter(Z[i, 0], Z[i, 1], s=90, color=col, label=lb, edgecolors="white", linewidths=1)
                for i, v in enumerate(cl):
                    ax.annotate(str(v), Z[i], xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
                ax.legend(frameon=False, fontsize=8)
                ax.set_title(f"{name}: the 12 classes of n mod 12", fontsize=9, loc="left")
            else:
                for dgt in range(10):
                    i = cl % 10 == dgt; ax.scatter(Z[i, 0], Z[i, 1], s=45, color=CAT[dgt], label=f"ends in {dgt}", edgecolors="white", linewidths=0.6)
                ax.legend(frameon=False, fontsize=7, ncol=2)
                ax.set_title(f"{name}: the 60 classes of n mod 60", fontsize=9, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig("results/e11_shape_search/fig_mod_clusters.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
