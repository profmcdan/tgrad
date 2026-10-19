"""Directed-cycle structure of the Elliptic graph and the benchmark (seed 42): strongly
connected components, reciprocal pairs, tr(A^L) for L = 2..6, and the share of labelled
entities on a directed cycle. Run from the repository root; needs data/elliptic_bitcoin_dataset/."""
import numpy as np, pandas as pd, scipy.sparse as sp, json, sys, os
from scipy.sparse.csgraph import connected_components
sys.path.append("src")

def stats(name, src, dst, n, label):
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    A.data[:] = 1.0; A.sum_duplicates(); A.data[:] = 1.0
    m = A.nnz
    selfloops = int(A.diagonal().sum())
    recip = int(A.multiply(A.T).nnz // 2)                 # unordered reciprocal pairs = 2-cycles
    ncomp, lab = connected_components(A, directed=True, connection="strong")
    sizes = np.bincount(lab)
    nontriv = sizes >= 2
    on_cycle = nontriv[lab]                               # node lies in an SCC of size >= 2
    out = {"dataset": name, "nodes": int(n), "edges": int(m), "self_loops": selfloops,
           "reciprocal_pairs": recip, "scc_count": int(ncomp),
           "nontrivial_scc_count": int(nontriv.sum()),
           "largest_scc": int(sizes.max()),
           "nodes_on_any_directed_cycle": int(on_cycle.sum()),
           "frac_nodes_on_any_directed_cycle": float(on_cycle.mean())}
    # closed walks tr(A^L): exactly zero outside non-trivial SCCs, so restrict to them
    idx = np.where(on_cycle)[0]
    tr = {}
    if len(idx):
        S = A[idx][:, idx].tocsr(); P = S.copy()
        for L in range(2, 7):
            if L > 2 or True:
                if L == 2: P = S @ S
                else: P = P @ S
            tr[L] = float(P.diagonal().sum())
    else:
        tr = {L: 0.0 for L in range(2, 7)}
    out["closed_walks_trace"] = tr
    # per-class: labelled nodes on a directed cycle
    for cls, nm in [(1, "illicit"), (0, "licit")]:
        sel = label == cls
        out[f"{nm}_labelled"] = int(sel.sum())
        out[f"{nm}_on_cycle"] = int((on_cycle & sel).sum())
        out[f"{nm}_frac_on_cycle"] = float((on_cycle & sel).sum() / max(1, sel.sum()))
    return out

res = {}
# ---- Elliptic ----
d = "data/elliptic_bitcoin_dataset"
feat = pd.read_csv(f"{d}/elliptic_txs_features.csv", header=None)
ids = feat[0].values; tstep = feat[1].values
cls = pd.read_csv(f"{d}/elliptic_txs_classes.csv")
edges = pd.read_csv(f"{d}/elliptic_txs_edgelist.csv")
pos = {v: i for i, v in enumerate(ids)}
src = edges.iloc[:, 0].map(pos).values; dst = edges.iloc[:, 1].map(pos).values
cmap = dict(zip(cls.iloc[:, 0], cls.iloc[:, 1].astype(str)))
label = np.array([1 if cmap.get(v) == "1" else (0 if cmap.get(v) == "2" else -1) for v in ids])
res["elliptic"] = stats("Elliptic", src, dst, len(ids), label)
# temporal direction of Elliptic edges: do edges ever point backwards in time?
ts_s, ts_d = tstep[src], tstep[dst]
res["elliptic"]["edges_same_timestep"] = float((ts_s == ts_d).mean())
res["elliptic"]["edges_backward_in_time"] = int((ts_d < ts_s).sum())

# ---- controlled benchmark (published seed) ----
from tgrad.data.dataset import SyntheticCyclicAML
ds = SyntheticCyclicAML(num_nodes=6000, feat_dim=165, num_benign_edges=18000, num_cycles=600, seed=42)
ds.load_data()
ei = ds.edge_index.numpy()
res["benchmark"] = stats("Cyclic benchmark", ei[0], ei[1], ds.num_nodes, ds.node_label_t.numpy())

for k, v in res.items():
    print(f"\n=== {v['dataset']} ===")
    for kk, vv in v.items():
        if kk != "dataset": print(f"  {kk:34} {vv}")
json.dump(res, open("results/structure.json", "w"), indent=2)
