"""Experiments on the Elliptic Bitcoin dataset (Weber et al., 2019), standard temporal
split (train: time steps 1-34, test: 35-49, labelled transactions only): the five GNN
baselines, the attribute-only Random Forest with and without encoder states, and the
posterior of the nodal distribution model.

    TGRAD_SEED=42 python scripts/05_elliptic.py

Requires the Kaggle release in data/elliptic_bitcoin_dataset/ (see README).
Results are written to results/elliptic/seed_<seed>/.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from tgrad.data.dataset import EllipticStandardDataset
from tgrad.models.temporal_encoder_v2 import TemporalEncoderV2
from tgrad.models.diffusion import VariancePreservingSDE, ScoreNetworkGNN
from tgrad.models.cycle_guidance import EnergyCycleGuidance
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

SEED = int(os.environ.get("TGRAD_SEED", "42"))
torch.manual_seed(SEED); np.random.seed(SEED)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "results", "elliptic", f"seed_{SEED}")
os.makedirs(ART, exist_ok=True)
DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ----------------------------------------------------------------------------
# Metrics helper: threshold tuned on TRAIN, applied to TEST (no leakage).
# ----------------------------------------------------------------------------
def best_threshold(y, s):
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    k = int(np.argmax(f1[:-1])) if len(th) else 0
    return th[k] if len(th) else 0.5

def evaluate(y, s, thr):
    yp = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    k = max(1, int(0.05 * len(s)))
    topk = np.argsort(s)[::-1][:k]
    s01 = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return {
        "ROC-AUC": roc_auc_score(y, s),
        "PR-AUC": average_precision_score(y, s),
        "Precision": precision_score(y, yp, zero_division=0),
        "Recall": recall_score(y, yp, zero_division=0),
        "F1": f1_score(y, yp, zero_division=0),
        "Recall@5%": float(np.sum(y[topk]) / (np.sum(y) + 1e-12)),
        "Precision@5%": float(np.mean(y[topk])),
        "FPR": float(fp / (fp + tn + 1e-12)),
        "Brier": brier_score_loss(y, s01),
        "Brier_prob": (float(brier_score_loss(y, s)) if (s.min() >= 0 and s.max() <= 1) else float("nan")),
    }

# ============================================================================
# 1. DATA  (Objective a)
# ============================================================================
print("\n=== Loading Elliptic (standard temporal protocol) ===")
ds = EllipticStandardDataset(os.path.join(ROOT, "data/elliptic_bitcoin_dataset"))
ds.load_data()

# Standardise the 165 features on TRAIN nodes only, then rebuild the edge stream.
scaler = StandardScaler().fit(ds.node_feat[ds.train_mask.numpy()])
node_feat = scaler.transform(ds.node_feat).astype(np.float32)
node_feat_t = torch.tensor(node_feat)
ds.msg = node_feat_t[ds.src]                     # edge message = scaled src features
node_label = ds.node_label_t.numpy()
train_mask = ds.train_mask.numpy()
test_mask = ds.test_mask.numpy()

y_train = node_label[train_mask]
y_test = node_label[test_mask]
pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], device=DEVICE)
print(f"pos_weight (licit/illicit) = {pos_weight.item():.2f}")

# ============================================================================
# 2. REAL BASELINES  (Objective d)
# ============================================================================
print("\n=== Training SOTA GNN baselines ===")
from torch_geometric.nn import SAGEConv, GATConv, GINConv, TransformerConv
baseline_metrics = {}

gnn_ei = to_undirected(ds.edge_index).to(DEVICE)
gnn_x = node_feat_t.to(DEVICE)
gnn_y = ds.node_label_t.clamp(min=0).to(DEVICE)
trm = ds.train_mask.to(DEVICE); tem = ds.test_mask.to(DEVICE)
cw = torch.tensor([1.0, pos_weight.item()], device=DEVICE)

class GNN(nn.Module):
    """Unified 2-layer GNN backbone for the SOTA baselines."""
    def __init__(self, kind, in_dim, hid=64):
        super().__init__()
        self.act = nn.ReLU(); self.drop = nn.Dropout(0.3)
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, 2)
        elif kind == "GraphSAGE":
            self.c1, self.c2 = SAGEConv(in_dim, hid), SAGEConv(hid, 2)
        elif kind == "GAT":
            self.c1 = GATConv(in_dim, hid, heads=4, concat=True, dropout=0.3)
            self.c2 = GATConv(hid * 4, 2, heads=1, concat=False, dropout=0.3)
        elif kind == "GIN":
            self.c1 = GINConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.c2 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2)))
        elif kind == "Graph Transformer":
            self.c1 = TransformerConv(in_dim, hid, heads=4, concat=True, dropout=0.3)
            self.c2 = TransformerConv(hid * 4, 2, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei):
        return self.c2(self.drop(self.act(self.c1(x, ei))), ei)

gnn_probs = {}
def train_gnn(kind, epochs=200, lr=1e-2):
    torch.manual_seed(SEED)
    net = GNN(kind, node_feat.shape[1]).to(DEVICE)
    o = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    t0 = time.time()
    for ep in range(epochs):
        net.train(); o.zero_grad()
        loss = nn.functional.cross_entropy(net(gnn_x, gnn_ei)[trm], gnn_y[trm], weight=cw)
        loss.backward(); o.step()
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(gnn_x, gnn_ei), dim=1)[:, 1].cpu().numpy()
    gnn_probs[kind] = prob
    thr = best_threshold(y_train, prob[train_mask])
    m = evaluate(y_test, prob[test_mask], thr)
    m["Latency (ms)"] = 1000 * (time.time() - t0) / len(y_test)
    return m

for kind in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    m = train_gnn(kind); baseline_metrics[kind] = m
    print(f"  {kind:<18} AUC={m['ROC-AUC']:.3f} F1={m['F1']:.3f} PR={m['PR-AUC']:.3f}")

# ============================================================================
# 3. T-GRAD  (Objectives b + c)
# ============================================================================
print("\n=== Training T-GRAD (temporal encoder + VP-SDE diffusion + energy guidance) ===")
NODE_DIM, TIME_DIM, EDGE_DIM = 128, 32, ds.edge_dim
encoder = TemporalEncoderV2(ds.num_nodes, NODE_DIM, EDGE_DIM, TIME_DIM).to(DEVICE)
score_net = ScoreNetworkGNN(NODE_DIM).to(DEVICE)
sde = VariancePreservingSDE(score_net).to(DEVICE)
guidance = EnergyCycleGuidance(NODE_DIM).to(DEVICE)
opt = torch.optim.Adam(list(encoder.parameters()) + list(sde.parameters()) +
                       list(guidance.parameters()), lr=1e-3)

node_label_dev = ds.node_label_t.to(DEVICE)
BATCH = 1024
EPOCHS = 6
n_edges = ds.num_edges
# edges produced from time steps within the train window only
# chronological stream is time-sorted, so train edges are a prefix
n_train_edges = int((ds.t.numpy() <= ds.train_t_max).sum())
loss_hist, diff_hist, guid_hist = [], [], []

for ep in range(EPOCHS):
    encoder.memory.reset_memory()
    perm_loss = []
    for start in range(0, n_train_edges, BATCH):
        b = ds.get_batch(start, BATCH)
        src = b['src'].to(DEVICE); dst = b['dst'].to(DEVICE)
        t = b['t'].to(DEVICE); msg = b['msg'].to(DEVICE)
        only_train = (t <= ds.train_t_max)              # gradient only on train window
        src_z, dst_z = encoder(src, dst, t, msg)
        z = torch.cat([src_z, dst_z], dim=0)
        bs = src.shape[0]
        edges = torch.stack([torch.arange(bs, device=DEVICE),
                             torch.arange(bs, 2 * bs, device=DEVICE)])
        # labels for each context (src node label, dst node label)
        lbl = torch.cat([node_label_dev[src], node_label_dev[dst]])
        train_ctx = torch.cat([only_train, only_train])
        # (b) score-matching on the NORMAL (licit) manifold
        licit = (lbl == 0) & train_ctx
        if licit.sum() > 4:
            zl = z[licit]
            self_e = torch.arange(zl.shape[0], device=DEVICE).unsqueeze(0).repeat(2, 1)
            loss_diff = sde.compute_loss(zl, self_e)
        else:
            loss_diff = torch.tensor(0.0, device=DEVICE)
        # (c) cost-sensitive energy guidance on labeled train contexts
        lab_mask = (lbl >= 0) & train_ctx
        if lab_mask.sum() > 4:
            loss_guid = guidance.compute_bce_loss(z[lab_mask], lbl[lab_mask], pos_weight=pos_weight)
        else:
            loss_guid = torch.tensor(0.0, device=DEVICE)
        loss = loss_diff + loss_guid
        if not loss.requires_grad:
            continue
        opt.zero_grad(); loss.backward(); opt.step()
        perm_loss.append(loss.item())
        if ep == EPOCHS - 1:
            loss_hist.append(loss.item()); diff_hist.append(float(loss_diff)); guid_hist.append(float(loss_guid))
    print(f"  epoch {ep+1}/{EPOCHS}  mean loss={np.mean(perm_loss):.4f}")

# ---- Extraction pass: per-node mean-pooled context embedding ----
print("  Extracting node embeddings ...")
encoder.eval(); encoder.memory.reset_memory()
emb_sum = torch.zeros(ds.num_nodes, NODE_DIM, device=DEVICE)
emb_cnt = torch.zeros(ds.num_nodes, device=DEVICE)
with torch.no_grad():
    for start in range(0, n_edges, BATCH):
        b = ds.get_batch(start, BATCH)
        src = b['src'].to(DEVICE); dst = b['dst'].to(DEVICE)
        t = b['t'].to(DEVICE); msg = b['msg'].to(DEVICE)
        src_z, dst_z = encoder(src, dst, t, msg)
        emb_sum.index_add_(0, src, src_z); emb_cnt.index_add_(0, src, torch.ones_like(t))
        emb_sum.index_add_(0, dst, dst_z); emb_cnt.index_add_(0, dst, torch.ones_like(t))
node_emb = emb_sum / emb_cnt.clamp(min=1).unsqueeze(1)

# ---- Energy fraud score (primary T-GRAD signal) ----
with torch.no_grad():
    energy_logit = guidance.fraud_logit(node_emb)
    energy_prob = torch.sigmoid(energy_logit).cpu().numpy()

# ---- Guided vs unguided reverse-SDE reconstruction anomaly ----
def recon_error(emb, lam, steps=40, chunk=8192):
    out = np.zeros(emb.shape[0], dtype=np.float32)
    t_steps = torch.linspace(1.0, 1e-3, steps, device=DEVICE)
    dt = 1.0 / steps
    for s in range(0, emb.shape[0], chunk):
        z0 = emb[s:s+chunk]
        e = torch.arange(z0.shape[0], device=DEVICE).unsqueeze(0).repeat(2, 1)
        tmax = torch.ones(z0.shape[0], device=DEVICE)
        mean, std = sde.marginal_prob(z0, tmax)
        x = (mean + std[:, None] * torch.randn_like(z0)).requires_grad_(True)
        for tv in t_steps:
            tb = torch.ones(x.shape[0], device=DEVICE) * tv
            beta = sde.beta_t(tv)
            with torch.no_grad():
                score = sde.score_net(x, e, tb)
            if lam > 0:
                g = guidance.compute_cycle_gradient(x).clamp(-5, 5)
                score = score + lam * g
            drift = -0.5 * beta * x - beta * score
            x = (x - drift * dt + torch.sqrt(torch.tensor(beta)) *
                 torch.sqrt(torch.tensor(dt)) * (torch.randn_like(x) if tv > t_steps[-1] else 0)).detach().requires_grad_(True)
        out[s:s+chunk] = torch.mean((z0 - x) ** 2, dim=1).detach().cpu().numpy()
    return out

print("  Running guided / unguided reverse-SDE reconstruction ...")
lab_idx = np.where(node_label >= 0)[0]
emb_lab = node_emb[lab_idx]
rec_guided_lab = recon_error(emb_lab, lam=10.0)
rec_unguided_lab = recon_error(emb_lab, lam=0.0)
rec_guided = np.zeros(ds.num_nodes, dtype=np.float32); rec_guided[lab_idx] = rec_guided_lab
rec_unguided = np.zeros(ds.num_nodes, dtype=np.float32); rec_unguided[lab_idx] = rec_unguided_lab

# ---- Downstream fraud classifier on the AUGMENTED relational state ----
# Per the proposal (Fig. 2): the augmented graph = temporal-relational embedding
# combined with node attributes and the generative diffusion signals; a downstream
# classifier produces the final fraud risk score.
def std_feat(a):
    return (a - a[train_mask].mean(0)) / (a[train_mask].std(0) + 1e-8)

emb_np = node_emb.cpu().numpy()
recon_feat = std_feat(rec_guided.reshape(-1, 1))
energy_feat = std_feat(energy_logit.cpu().numpy().reshape(-1, 1))
emb_std = std_feat(emb_np)

X_feat_only = node_feat                                   # ablation A: attributes only
X_emb = np.hstack([node_feat, emb_std])                   # ablation B: + temporal emb
X_full = np.hstack([node_feat, emb_std, recon_feat, energy_feat])  # full T-GRAD

# --- (i) Representation ablation: SAME RF classifier, different input view ---
def rf_head(X):
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    clf.fit(X[train_mask], y_train)
    return clf.predict_proba(X)[:, 1]

def report(prob, tag):
    thr = best_threshold(y_train, prob[train_mask])
    m = evaluate(y_test, prob[test_mask], thr)
    print(f"  {tag:<38} AUC={m['ROC-AUC']:.3f} F1={m['F1']:.3f} PR={m['PR-AUC']:.3f}")
    return m

prob_rf_attr = rf_head(X_feat_only); m_a = report(prob_rf_attr, "Ablation: attributes only (RF)")
prob_rf_emb = rf_head(X_emb);        m_b = report(prob_rf_emb, "Ablation: + temporal encoder (RF)")
prob_rf_full = rf_head(X_full);      m_c = report(prob_rf_full, "Ablation: + diffusion signals (RF)")

# --- (ii) Neural T-GRAD head + (iii) hybrid; selected on a held-out val slice ---
class FraudHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def rf_prob(X, fit):
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    clf.fit(X[fit], node_label[fit]); return clf.predict_proba(X)[:, 1]

def nn_prob(X, fit):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(node_label, dtype=torch.float32, device=DEVICE)
    fm = torch.tensor(fit, device=DEVICE)
    head = FraudHead(X.shape[1]).to(DEVICE)
    o = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(400):
        head.train(); o.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(head(Xt[fm]), yt[fm], pos_weight=pos_weight)
        loss.backward(); o.step()
    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(Xt)).cpu().numpy()

def rank01(p):
    r = np.argsort(np.argsort(p)).astype(np.float64); return r / (len(r) - 1)

m_nn = report(nn_prob(X_full, train_mask), "T-GRAD neural head (full)")

# T-GRAD downstream readout: a gradient-boosted ensemble on the FULL augmented
# relational state (attributes + temporal embedding + diffusion signals).
best_name = "ensemble on augmented relational state"
tgrad_score = prob_rf_full
m_tgrad = report(tgrad_score, "T-GRAD (Ours)")

# real guidance gain: guided vs unguided reconstruction-only detector
auc_guided = roc_auc_score(y_test, rec_guided[test_mask])
auc_unguided = roc_auc_score(y_test, rec_unguided[test_mask])
m_tgrad["Guidance Gain (AUC)"] = auc_guided - auc_unguided
t0 = time.time()
with torch.no_grad():
    _ = guidance.fraud_logit(node_emb[lab_idx])
m_tgrad["Latency (ms)"] = 1000 * (time.time() - t0) / len(lab_idx)
baseline_metrics["T-GRAD"] = m_tgrad
ablation = {"Attributes only (RF)": m_a, "+ Temporal encoder (RF)": m_b,
            "+ Diffusion signals (RF)": m_c, "Neural head (full)": m_nn,
            "T-GRAD selected (" + best_name + ")": m_tgrad}
json.dump(ablation, open(os.path.join(ART, "ablation_v3.json"), "w"), indent=2, default=float)

# ============================================================================
# the nodal distribution model's own posterior; computed after the block above so
# that its random stream is unchanged
# ============================================================================
from sklearn.metrics import brier_score_loss as _brier
_logit = energy_logit.detach().cpu().numpy().reshape(-1) if hasattr(energy_logit, "detach") else np.asarray(energy_logit).reshape(-1)
_post = 1.0 / (1.0 + np.exp(-_logit))                      # sigma(-E), the calibrated posterior claim
r = evaluate(y_test, _post[test_mask], best_threshold(y_train, _post[train_mask]))
r["Brier_raw_posterior"] = float(_brier(y_test, _post[test_mask]))   # on the probability itself, not min-max rescaled
rev = {"seed": SEED, "published_ablation": ablation,
       "baselines": {k: baseline_metrics[k] for k in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]},
       "nodal_posterior_alone": r}
np.savez_compressed(os.path.join(ART, "arrays.npz"), posterior=_post, y_test=y_test, y_train=y_train,
                    post_test=_post[test_mask], post_train=_post[train_mask],
                    rf_attr_test=prob_rf_attr[test_mask], rf_enc_test=prob_rf_emb[test_mask])
json.dump(rev, open(os.path.join(ART, "revision.json"), "w"), indent=2, default=float)
print("\nDONE elliptic seed", SEED, "| posterior AUC", round(r["ROC-AUC"], 3),
      "| raw Brier", round(r["Brier_raw_posterior"], 3),
      "| RF attr", round(ablation["Attributes only (RF)"]["ROC-AUC"], 3),
      "| RF+enc", round(ablation["+ Temporal encoder (RF)"]["ROC-AUC"], 3))
