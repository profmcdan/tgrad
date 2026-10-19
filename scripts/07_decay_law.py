"""Proposition 2 in practice: attenuation of the closed-walk counts under independent edge
loss, against the predictions q^L and kappa^L, and detection by the raw monitor for the
binary, oracle and two-hop-candidate variants (benchmark, seed 42, five draws per q).

    python scripts/07_decay_law.py            # recompute, then plot
    python scripts/07_decay_law.py --plot     # plot from results/decay_law.json
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
from tgrad.data.dataset import SyntheticCyclicAML        # noqa: E402
from monitor_utils import closed_walks, monitor, two_hop_candidates  # noqa: E402

LENGTHS = (3, 4, 5)
Q_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
P_HATS = (0.25, 0.5)
REPEATS = 5


def deflection(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical deflection coefficient (m1 - m0)^2 / mean variance."""
    a, b = x[y == 1], x[y == 0]
    if a.size < 2 or b.size < 2:
        return float("nan")
    v = 0.5 * (a.var() + b.var())
    return float((a.mean() - b.mean()) ** 2 / v) if v > 0 else float("nan")


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def run_prop1(ds, seed=42):
    ei = ds.edge_index.numpy()
    N = int(ds.num_nodes)
    lab = ds.node_label_t.numpy()
    sel = lab >= 0                      # Elliptic has unknown labels

    ones = np.ones(ei.shape[1])
    A_full = sp.coo_matrix((ones, (ei[0], ei[1])), shape=(N, N)).tocsr()
    full = {L: closed_walks(A_full, L) for L in LENGTHS}
    full_mean = {L: float(full[L].mean()) for L in LENGTHS}

    print(f"\n[Prop 2] SyntheticCyclicAML: {N} nodes, {ei.shape[1]} events, "
          f"{int((lab == 1).sum())} illicit")
    print(f"[Prop 2] fully observed mean (A^L)_ii: "
          + "  ".join(f"L={L}: {full_mean[L]:.4g}" for L in LENGTHS))

    rows = []
    rng = np.random.default_rng(seed)
    for q in Q_GRID:
        acc = {"bin": {L: [] for L in LENGTHS}}
        for p in P_HATS:
            acc[f"soft{p}"] = {L: [] for L in LENGTHS}
        names = ["bin"] + [f"soft{p}" for p in P_HATS] + ["cand"]
        auc_acc = {n: [] for n in names}
        auc3_acc = {n: [] for n in names}
        defl_acc = {n: [] for n in names}
        sep_acc = {n: [] for n in names}

        for _ in range(REPEATS):
            keep = rng.random(ei.shape[1]) < q
            src_o, dst_o = ei[0][keep], ei[1][keep]
            src_m, dst_m = ei[0][~keep], ei[1][~keep]

            A_bin = sp.coo_matrix((np.ones(keep.sum()), (src_o, dst_o)),
                                  shape=(N, N)).tocsr()
            variants = {"bin": A_bin}
            for p in P_HATS:
                # ORACLE arm: dropped events are restored at plausibility p.
                # This is exactly what Proposition 2 assumes -- a plausibility
                # model that is right about which pairs are missing -- so it
                # tests the decay law, not a deployable detector.
                w = np.concatenate([np.ones(keep.sum()),
                                    np.full((~keep).sum(), p)])
                r = np.concatenate([src_o, src_m])
                c = np.concatenate([dst_o, dst_m])
                variants[f"soft{p}"] = sp.coo_matrix((w, (r, c)),
                                                     shape=(N, N)).tocsr()
            # REALISTIC arm: plausibility p is spread over every two-hop
            # candidate pair, most of which are not genuinely missing edges.
            cr, cc = two_hop_candidates(A_bin)
            p = P_HATS[-1]
            w = np.concatenate([np.ones(keep.sum()), np.full(len(cr), p)])
            variants["cand"] = sp.coo_matrix(
                (w, (np.concatenate([src_o, cr]), np.concatenate([dst_o, cc]))),
                shape=(N, N)).tocsr()

            for name, A in variants.items():
                cw = {L: closed_walks(A, L) for L in LENGTHS}
                if name in acc:
                    for L in LENGTHS:
                        acc[name][L].append(float(cw[L].mean()))
                m = monitor(A)
                auc_acc[name].append(safe_auc(lab[sel], m[sel]))
                auc3_acc[name].append(safe_auc(lab[sel], cw[3][sel]))
                defl_acc[name].append(deflection(m[sel], lab[sel]))
                a3i = cw[3][sel][lab[sel] == 1].mean()
                a3l = cw[3][sel][lab[sel] == 0].mean()
                sep_acc[name].append(float(a3i / a3l) if a3l > 0 else float("nan"))

        row = {"q": q}
        for L in LENGTHS:
            row[f"pred_bin_L{L}"] = q ** L
            row[f"obs_bin_L{L}"] = (float(np.mean(acc["bin"][L])) / full_mean[L]
                                    if full_mean[L] else float("nan"))
            for p in P_HATS:
                k = q + (1 - q) * p
                row[f"pred_soft{p}_L{L}"] = k ** L
                row[f"obs_soft{p}_L{L}"] = (float(np.mean(acc[f"soft{p}"][L]))
                                            / full_mean[L] if full_mean[L] else float("nan"))
        for name in variants:
            row[f"auc_{name}"] = float(np.nanmean(auc_acc[name]))
            row[f"auc3_{name}"] = float(np.nanmean(auc3_acc[name]))
            row[f"defl_{name}"] = float(np.nanmean(defl_acc[name]))
            row[f"sep3_{name}"] = float(np.nanmean(sep_acc[name]))
        rows.append(row)
        print(f"  q={q:.1f}  bin L3 obs/pred={row['obs_bin_L3']:.3f}/{row['pred_bin_L3']:.3f}"
              f"  soft0.5 L3 obs/pred={row['obs_soft0.5_L3']:.3f}/{row['pred_soft0.5_L3']:.3f}"
              f"  monitor AUC bin={row['auc_bin']:.3f} oracle={row['auc_soft0.5']:.3f}"
              f" cand={row['auc_cand']:.3f}  S3 bin={row['sep3_bin']:.1f}")

    out = {"dataset": "SyntheticCyclicAML", "lengths": list(LENGTHS), "p_hats": list(P_HATS),
           "repeats": REPEATS, "full_mean_closed_walks": full_mean, "rows": rows}
    return out


def plot_prop1(res):
    rows = res["rows"]
    q = np.array([r["q"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    for L, c in zip(LENGTHS, ["#1f78b4", "#33a02c", "#6a3d9a"]):
        ax[0].plot(q, [r[f"pred_bin_L{L}"] for r in rows], "--", color=c, lw=1.6)
        ax[0].plot(q, [r[f"obs_bin_L{L}"] for r in rows], "o-", color=c, lw=2,
                   label=f"$L={L}$")
    ax[0].set_title(r"Binary monitor: observed vs predicted $q^{L}$")
    ax[0].set_xlabel("edge-retention probability $q$")
    ax[0].set_ylabel(r"mean $(A^L)_{ii}$ relative to full graph")

    p = P_HATS[-1]
    for L, c in zip(LENGTHS, ["#1f78b4", "#33a02c", "#6a3d9a"]):
        ax[1].plot(q, [r[f"pred_soft{p}_L{L}"] for r in rows], "--", color=c, lw=1.6)
        ax[1].plot(q, [r[f"obs_soft{p}_L{L}"] for r in rows], "s-", color=c, lw=2,
                   label=f"$L={L}$")
    ax[1].set_title(r"Weighted monitor ($\hat{p}=%.2f$): vs $\kappa^{L}$" % p)
    ax[1].set_xlabel("edge-retention probability $q$")

    ax[2].plot(q, [r["auc_bin"] for r in rows], "o-", color="#e31a1c", lw=2,
               label="binary (observed only)")
    ax[2].plot(q, [r[f"auc_soft{P_HATS[-1]}"] for r in rows], "s-", color="#1f78b4",
               lw=2, label=r"oracle plausibility $\hat{p}=%.2f$" % P_HATS[-1])
    ax[2].plot(q, [r["auc_cand"] for r in rows], "^-", color="#33a02c", lw=2,
               label=r"two-hop candidates $\hat{p}=%.2f$" % P_HATS[-1])
    ax[2].axhline(0.5, ls=":", color="grey")
    ax[2].set_title(r"Detection from the monitor $M_i$ under edge loss")
    ax[2].set_xlabel("edge-retention probability $q$"); ax[2].set_ylabel("ROC-AUC")

    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=9); a.invert_xaxis()
    fig.suptitle("Proposition 2: monitor degradation under independent edge loss "
                 "(benchmark, seed 42); dashed = predicted", y=1.02)
    fig.tight_layout()
    path = os.path.join(ROOT, "figures", "decay_law.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  wrote", path)




if __name__ == "__main__":
    path = os.path.join(ROOT, "results", "decay_law.json")
    if "--plot" in sys.argv:
        res = json.load(open(path))
    else:
        ds = SyntheticCyclicAML(seed=42); ds.load_data()
        res = run_prop1(ds)
        json.dump(res, open(path, "w"), indent=2)
    plot_prop1(res)
