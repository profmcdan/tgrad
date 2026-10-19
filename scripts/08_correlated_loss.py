"""Independent vs institution-correlated edge loss (Paper 1, table tab:corr).

Run from the repository root."""
import numpy as np, scipy.sparse as sp, sys, json
from sklearn.metrics import roc_auc_score
sys.path.append("src"); sys.path.append("scripts")
from tgrad.data.dataset import SyntheticCyclicAML
from monitor_utils import closed_walks, monitor

ds = SyntheticCyclicAML(num_nodes=6000, feat_dim=165, num_benign_edges=18000, num_cycles=600, seed=42); ds.load_data()
ei = ds.edge_index.numpy(); N = ds.num_nodes; lab = ds.node_label_t.numpy(); te = ds.test_mask.numpy()
def adj(keep):
    A = sp.csr_matrix((np.ones(keep.sum()), (ei[0, keep], ei[1, keep])), shape=(N, N)); A.data[:] = 1; return A
full = {L: closed_walks(adj(np.ones(ei.shape[1], bool)), L).mean() for L in (3, 5)}
m_full = monitor(adj(np.ones(ei.shape[1], bool)))
res = {"full_mean": full, "full_auc_all": roc_auc_score(lab, m_full), "full_auc_test": roc_auc_score(lab[te], m_full[te]), "rows": []}
rng = np.random.default_rng(0); DRAWS = 20
for q in (0.8, 0.6, 0.4):
    for mech, K in [("independent", None), ("institution", 30), ("institution", 5)]:
        att = {3: [], 5: []}; aucs = []; aucs_te = []
        for _ in range(DRAWS):
            if mech == "independent":
                keep = rng.random(ei.shape[1]) < q
            else:
                inst = rng.integers(0, K, N)                 # institution of each node
                obs = rng.random(K) < q                      # which institutions report
                keep = obs[inst[ei[0]]]                      # an edge is seen iff its source reports
            A = adj(keep)
            for L in (3, 5): att[L].append(closed_walks(A, L).mean() / full[L])
            m = monitor(A); aucs.append(roc_auc_score(lab, m)); aucs_te.append(roc_auc_score(lab[te], m[te]))
        row = {"q": q, "mechanism": mech, "K": K,
               "att3_mean": float(np.mean(att[3])), "att3_std": float(np.std(att[3])), "pred3": q**3,
               "att5_mean": float(np.mean(att[5])), "att5_std": float(np.std(att[5])), "pred5": q**5,
               "auc_all_mean": float(np.mean(aucs)), "auc_all_std": float(np.std(aucs)),
               "auc_test_mean": float(np.mean(aucs_te)), "auc_test_std": float(np.std(aucs_te))}
        res["rows"].append(row)
        print(f"q={q} {mech:11} K={str(K):4}  L3 obs {row['att3_mean']:.3f}±{row['att3_std']:.3f} pred {q**3:.3f} | "
              f"L5 obs {row['att5_mean']:.3f}±{row['att5_std']:.3f} pred {q**5:.3f} | AUC test {row['auc_test_mean']:.3f}±{row['auc_test_std']:.3f}")
print(f"\nfully observed: raw monitor AUC all-nodes {res['full_auc_all']:.3f} | test-split {res['full_auc_test']:.3f}")
json.dump(res, open("results/correlated_loss.json", "w"), indent=2)
