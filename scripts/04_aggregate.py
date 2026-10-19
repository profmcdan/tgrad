"""Mean and sample standard deviation over the five seeds of 03_benchmark.py.

    python scripts/04_aggregate.py _L6    # monitor L = 2..6 (the paper)
    python scripts/04_aggregate.py        # truncation check, L = 2..4
"""
import json, sys, os, numpy as np
suffix = sys.argv[1] if len(sys.argv) > 1 else ""          # "" = L<=4 truncation check, "_L6" = the paper
RD = "benchmark"
SEEDS = [42, 0, 1, 2, 3]
R = [json.load(open(f"results/{RD}/seed_{s}{suffix}/revision.json")) for s in SEEDS]
M = ["ROC-AUC", "PR-AUC", "F1", "Precision@5%", "FPR", "Brier_prob"]
def agg(get):
    out = {}
    for m in M:
        v = np.array([get(r).get(m, np.nan) for r in R], float)
        out[m] = (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan"))
    return out
table = {}
for grp in ["baselines_attr", "baselines_matched", "baselines_matched_walks_only", "ablation_factorial", "raw_scores"]:
    if grp not in R[0]: continue
    for k in R[0][grp]:
        if isinstance(R[0][grp][k], dict):
            table[f"{grp}:{k}"] = agg(lambda r, g=grp, kk=k: r[g][kk])
        else:
            v = np.array([r[grp][k] for r in R], float)
            table[f"{grp}:{k}"] = {"value": (float(v.mean()), float(v.std(ddof=1)))}
json.dump(table, open(f"results/{RD}/aggregate{suffix or '_L4'}.json", "w"), indent=2)
w = max(len(k) for k in table)
print(f"{'configuration':{w}}  " + "  ".join(f"{m:>13}" for m in M))
for k, v in table.items():
    if "value" in v:
        print(f"{k:{w}}  {v['value'][0]:.3f} ± {v['value'][1]:.3f}"); continue
    print(f"{k:{w}}  " + "  ".join(("  nan        " if not np.isfinite(v[m][0]) else f"{v[m][0]:.3f}±{v[m][1]:.3f}").rjust(13) for m in M))
