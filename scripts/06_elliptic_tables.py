"""Aggregate the Elliptic seeds of 05_elliptic.py into the Elliptic table (LaTeX) and draw the
calibration figure of the nodal posterior, as trained and after the prior-shift correction."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
R = "results/elliptic"
runs = sorted(glob.glob(f"{R}/seed_*/revision.json"))
E = [json.load(open(f)) for f in runs]; seeds = [e["seed"] for e in E]
M = ["ROC-AUC", "PR-AUC", "F1", "FPR", "Brier_prob"]
def agg(rows):
    out = {}
    for m in M:
        v = np.array([r[m] for r in rows], float)
        out[m] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"))
    return out
T = {}
for g in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    T[g] = agg([e["baselines"][g] for e in E])
T["RF, attributes"] = agg([e["published_ablation"]["Attributes only (RF)"] for e in E])
T["RF, attributes + encoder"] = agg([e["published_ablation"]["+ Temporal encoder (RF)"] for e in E])
T["Nodal posterior, alone"] = agg([e["nodal_posterior_alone"] for e in E])
# Proposition 1 in practice: class weighting w+ trains at a shifted prior; remove it
# from the logit (a training-time quantity only) and re-score calibration per seed
def corrected(seed):
    a = np.load(f"{R}/seed_{seed}/arrays.npz")
    y, p = a["y_test"], np.clip(a["post_test"], 1e-7, 1 - 1e-7); ytr = a["y_train"]
    w = (ytr == 0).sum() / (ytr == 1).sum()
    pc = 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(w))))
    return {"Brier_raw": brier_score_loss(y, p), "Brier_corr": brier_score_loss(y, pc),
            "mean_raw": float(p.mean()), "mean_corr": float(pc.mean()), "prev": float(y.mean()),
            "const": float(y.mean() * (1 - y.mean())), "w": float(w)}
CORR = [corrected(sd) for sd in seeds]
row_c = dict(T["Nodal posterior, alone"])
row_c["Brier_prob"] = (float(np.mean([c["Brier_corr"] for c in CORR])), float(np.std([c["Brier_corr"] for c in CORR], ddof=1)) if len(CORR) > 1 else float("nan"))
row_c["FPR"] = (float("nan"), float("nan")); row_c["F1"] = (float("nan"), float("nan"))
T["Nodal posterior, prior-corrected"] = row_c
def cell(v, m):
    a, s = v[m]
    if not np.isfinite(a): return "--"
    return ("$%.3f$" % a) if not np.isfinite(s) else ("$%.3f\\pm%.3f$" % (a, s))
rows = []
for k, v in T.items():
    if k == "RF, attributes": rows.append(r"\midrule")
    rows.append(k + " & " + " & ".join(cell(v, m) for m in M) + r"\\")
tab = (r"""\begin{table}[t]
\caption{Elliptic test window (time steps $35$--$49$): mean $\pm$ standard deviation over %d seeds. The Random Forest rows are attribute-based references, not graph baselines. ``Nodal posterior, alone'' scores each transaction by the posterior of the nodal model; ``prior-corrected'' applies the prior shift of Proposition 1. Brier is computed on each model's predicted probabilities. Lower is better for FPR and Brier.}
\label{tab:elliptic}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{p{2.2cm}ccccc}
\toprule
Model & ROC-AUC & PR-AUC & $F_1$ & FPR & Brier\\
\midrule
""" % len(E)) + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
open(f"{R}/tab_elliptic.tex", "w").write(tab)
summ = {"n_seeds": len(E), "seeds": seeds,
        "posterior_auc": "%.3f ± %.3f" % T["Nodal posterior, alone"]["ROC-AUC"],
        "posterior_brier": "%.3f ± %.3f" % T["Nodal posterior, alone"]["Brier_prob"],
        "rf_attr_auc": "%.3f ± %.3f" % T["RF, attributes"]["ROC-AUC"],
        "rf_enc_auc": "%.3f ± %.3f" % T["RF, attributes + encoder"]["ROC-AUC"],
        "table": {k: {m: v[m] for m in M} for k, v in T.items()}}
# calibration of the posterior itself (seed 42 arrays), against the forest on attributes + encoder
a = np.load(f"{R}/seed_42/arrays.npz")
y, post, rfe = a["y_test"], a["post_test"], a["rf_enc_test"]
fig, ax = plt.subplots(figsize=(5.2, 4.2))
_w = (a["y_train"] == 0).sum() / (a["y_train"] == 1).sum(); _pp = np.clip(post, 1e-7, 1 - 1e-7)
postc = 1 / (1 + np.exp(-(np.log(_pp / (1 - _pp)) - np.log(_w))))
for sc, nm, c in [(post, "nodal posterior, as trained", "#d62728"), (postc, "nodal posterior, prior-corrected", "#2ca02c"), (rfe, "RF, attributes + encoder", "#1f78b4")]:
    ft, mp = calibration_curve(y, sc, n_bins=10, strategy="quantile")
    ax.plot(mp, ft, "o-", color=c, lw=1.8, ms=4, label=f"{nm} (Brier {brier_score_loss(y, sc):.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed illicit frequency")
ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9); plt.tight_layout()
plt.savefig("figures/elliptic_posterior_calibration.png", dpi=300); plt.close()
summ["seed42_post_range"] = [float(post.min()), float(post.max())]
summ["seed42_post_auc"] = float(roc_auc_score(y, post)); summ["seed42_post_brier"] = float(brier_score_loss(y, post))
summ["seed42_rfe_brier"] = float(brier_score_loss(y, rfe)); summ["test_prevalence"] = float(y.mean())
summ["corr"] = CORR
summ["corr_mean"] = {k: float(np.mean([c[k] for c in CORR])) for k in CORR[0]}
summ["seed42_post_train_auc"] = float(roc_auc_score(a["y_train"], a["post_train"]))
summ["summary"] = ("the nodal posterior alone reaches ROC-AUC %s against %s for the Random Forest on attributes"
                   % (summ["posterior_auc"], summ["rf_attr_auc"]))
json.dump(summ, open(f"{R}/aggregate_elliptic.json", "w"), indent=2)
print(json.dumps({k: v for k, v in summ.items() if k != "table"}, indent=1))
for k, v in T.items(): print(f"{k:28}", "  ".join("%s %.3f±%.3f" % (m, *v[m]) for m in M))
# ---- timing table
if os.path.exists("results/timing.json"):
    TM = json.load(open("results/timing.json")); tr = []
    for r in TM["rows"]:
        tr.append("%s & %s & %.1f & %s & %.3f & %.2f\\\\" % (f"{r['edges']:,}".replace(",", "{,}"), f"{r['nodes']:,}".replace(",", "{,}"),
                  r["encoder_sec"], f"{r['encoder_events_per_sec']:,.0f}".replace(",", "{,}"), r["energy_head_sec"], r["monitor_sec"]))
    hw = TM["hardware"]
    tt = (r"""\begin{table}[t]
\caption{Cost measured separately for each stage on AMLSim-style graphs of increasing size (%s, %s cores, %s\,GB, CPU only). Encoder: one inference pass over the event stream. Energy head: one forward pass per entity. Monitor: \eqref{eq:monitor} with $L\le6$ on the full adjacency. Wall-clock seconds, idle machine.}
\label{tab:timing}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{rrcccc}
\toprule
Edges & Entities & Encoder (s) & Events/s & Head (s) & Monitor (s)\\
\midrule
""" % (hw["cpu"], hw["cores"], hw["ram_gb"])) + "\n".join(tr) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    open("results/tab_timing.tex", "w").write(tt); print("\ntiming table written"); print(tt)
