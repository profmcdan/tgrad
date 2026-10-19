"""How many of the closed L-walks counted by
the monitor are simple directed cycles, and how many use distinct edges (the case covered by
Proposition 2). Exact enumeration on the benchmark graphs, L = 2..6, five seeds."""
import os, sys, json, io, contextlib
import numpy as np, scipy.sparse as sp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
from tgrad.data.dataset import SyntheticCyclicAML

LMAX = 6
def closed_walks(adj, nodes, L):
    """Enumerate closed L-walks from each start node; return per-node counts by kind."""
    tot = {}; simple = {}; dist = {}
    for s in nodes:
        stack = [(s, [s], [])]
        while stack:
            v, path, edges = stack.pop()
            if len(edges) == L:
                continue
            for w in adj[v]:
                e = (v, w); ne = edges + [e]
                if len(ne) == L:
                    if w == s:
                        tot[s] = tot.get(s, 0) + 1
                        if len(set(path)) == L: simple[s] = simple.get(s, 0) + 1
                        if len(set(ne)) == L: dist[s] = dist.get(s, 0) + 1
                else:
                    stack.append((w, path + [w], ne))
    return tot, simple, dist

out = {}
for seed in [42, 0, 1, 2, 3]:
    with contextlib.redirect_stdout(io.StringIO()):
        ds = SyntheticCyclicAML(num_nodes=6000, feat_dim=165, num_benign_edges=18000, num_cycles=600, seed=seed); ds.load_data()
    ei = ds.edge_index.numpy(); n = ds.num_nodes; y = ds.node_label_t.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)); A.data[:] = 1   # binary
    diag = {}; P = A
    for L in range(2, LMAX + 1):
        P = P @ A; diag[L] = P.diagonal()
    onwalk = np.where(sum(diag.values()) > 0)[0]
    keep = np.zeros(n, bool); keep[onwalk] = True
    adj = {v: [w for w in A.indices[A.indptr[v]:A.indptr[v + 1]] if keep[w]] for v in onwalk}
    res = {}
    for L in range(2, LMAX + 1):
        tot, simple, dist = closed_walks(adj, onwalk, L)
        T = sum(tot.values()); S = sum(simple.values()); D = sum(dist.values())
        assert T == int(diag[L].sum()), (T, diag[L].sum())
        cls = {}
        for c in (0, 1):
            idx = [i for i in onwalk if y[i] == c]
            t_ = sum(tot.get(i, 0) for i in idx); s_ = sum(simple.get(i, 0) for i in idx)
            cls[c] = {"walks": t_, "simple": s_}
        res[L] = {"trace": T, "simple_walks": S, "cycles": S // L, "distinct_edge_walks": D,
                  "frac_simple": S / T if T else None, "frac_distinct": D / T if T else None,
                  "licit": cls[0], "illicit": cls[1]}
    out[seed] = res
    print(seed, {L: (r["trace"], r["simple_walks"], r["distinct_edge_walks"]) for L, r in res.items()}, flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "walks_vs_cycles.json"), "w"), indent=1)
