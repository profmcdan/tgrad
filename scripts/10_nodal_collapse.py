"""Encoder-state collapse statistics and the nodal-logit figure (seed 42, L <= 6). Run from the repository root."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
RD = "results/benchmark/seed_42_L6"
z = np.load(f"{RD}/embeddings.npz")["node_emb"]; a = np.load(f"{RD}/arrays.npz")
zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
rng = np.random.default_rng(0); i = rng.integers(0, len(z), 20000); j = rng.integers(0, len(z), 20000)
cos = (zn[i] * zn[j]).sum(1)
mu = z.mean(0); share = float(np.linalg.norm(mu) ** 2 / (np.linalg.norm(z, axis=1) ** 2).mean())
el, lab, tr, te = a["energy_logit"], a["node_label"], a["train_mask"], a["test_mask"]
out = {"pairwise_cosine_mean": float(cos.mean()), "pairwise_cosine_sd": float(cos.std()),
       "share_sq_norm_mean_direction": share,
       "logit_range_train": float(np.ptp(el[tr])), "logit_range_test": float(np.ptp(el[te])),
       "logit_auc_train": float(roc_auc_score(lab[tr], el[tr])), "logit_auc_test": float(roc_auc_score(lab[te], el[te]))}
json.dump(out, open("results/nodal_collapse_seed42.json", "w"), indent=2); print(out)
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
bins = np.linspace(np.percentile(el, 0.5), np.percentile(el, 99.5), 45)
for k, (m, nm) in enumerate([(tr, "Training split"), (te, "Test split")]):
    ax[k].hist(el[m & (lab == 0)], bins=bins, density=True, alpha=0.55, color="#1f78b4", label="licit, $y=0$")
    ax[k].hist(el[m & (lab == 1)], bins=bins, density=True, alpha=0.55, color="#e31a1c", label="illicit, $y=1$")
    ax[k].set_title(f"{nm}   (ROC-AUC {roc_auc_score(lab[m], el[m]):.3f})", fontsize=11)
    ax[k].set_xlabel(r"fraud logit $f(\mathbf{z})$"); ax[k].grid(alpha=0.3)
ax[0].set_ylabel("density"); ax[0].legend(frameon=False)
plt.tight_layout(); plt.savefig("figures/nodal_logit_train_test.png", dpi=300); plt.close()
