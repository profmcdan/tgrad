"""Fill of A^L, L = 1..6, on a 200,000-edge benchmark-style graph: why the monitor is computed
from A^2 and A^3 only. Run from the repository root."""
import numpy as np, scipy.sparse as sp, sys, time, json
sys.path.append("src"); from tgrad.data.dataset import SyntheticCyclicAML
E = 200000; n = int(round(E / 3.45)); nc = int(round(0.13 * E / 4.5)); nb = int(round(E - nc * 4.5))
ds = SyntheticCyclicAML(num_nodes=n, feat_dim=8, num_benign_edges=nb, num_cycles=nc, cycle_len_range=(3, 6), time_span=100.0, seed=42); ds.load_data()
ei = ds.edge_index.numpy(); A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)); A.data[:] = 1
P = A; out = {}
for L in range(1, 7):
    if L > 1: P = P @ A
    out[L] = int(P.nnz); print(f"L={L}: nnz(A^L) = {P.nnz:,}  (~{P.nnz*12/2**30:.2f} GB in CSR)", flush=True)
    if P.nnz > 4e8: print("stopping: guard"); break
json.dump({"edges": int(ei.shape[1]), "nodes": n, "nnz": out}, open("results/fill_growth_200k.json", "w"), indent=1)
