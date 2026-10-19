"""Experiments on the controlled cyclic benchmark: GNN baselines with and without the
monitor's features, the component ablation, the raw monitor and degree scores, and
(with TGRAD_ARTIFACT=1) the learned-plausibility test.

Fraud is defined purely by directed-cycle topology; node attributes carry no information.

    TGRAD_SEED=42 TGRAD_MAXL=6 TGRAD_ARTIFACT=1 python scripts/03_benchmark.py

TGRAD_MAXL=6 gives the monitor of the paper (L = 2..6); TGRAD_MAXL=4 is the truncation
check. Results are written to results/benchmark/seed_<seed>[_L6]/revision.json.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss)

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from tgrad.data.dataset import SyntheticCyclicAML
from tgrad.models.temporal_encoder_v2 import TemporalEncoderV2
from tgrad.models.diffusion import VariancePreservingSDE, ScoreNetworkGNN
from tgrad.models.cycle_guidance import EnergyCycleGuidance
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

SEED = int(os.environ.get("TGRAD_SEED", "42"))
torch.manual_seed(SEED); np.random.seed(SEED)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "results", "benchmark", f"seed_{SEED}" + ("" if os.environ.get("TGRAD_MAXL", "4") == "4" else "_L" + os.environ["TGRAD_MAXL"]))
os.makedirs(ART, exist_ok=True)
DEVICE = "cpu"
C_GRID = 0.3

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
    return {"ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
            "Precision": precision_score(y, yp, zero_division=0),
            "Recall": recall_score(y, yp, zero_division=0),
            "F1": f1_score(y, yp, zero_division=0),
            "Recall@5%": float(np.sum(y[topk]) / (np.sum(y) + 1e-12)),
            "Precision@5%": float(np.mean(y[topk])),
            "FPR": float(fp / (fp + tn + 1e-12)),
            "Brier": brier_score_loss(y, s01),
            # proper Brier on the model's own probabilities; undefined for raw scores
            "Brier_prob": (float(brier_score_loss(y, s)) if (s.min() >= 0 and s.max() <= 1) else float("nan"))}

# ============================================================================
# 1. DATA
# ============================================================================
print("=== Building AMLSim-style cyclic dataset ===")
ds = SyntheticCyclicAML(num_nodes=6000, feat_dim=165, num_benign_edges=18000,
                        num_cycles=600, seed=SEED)
ds.load_data()
node_label = ds.node_label_t.numpy()
train_mask = ds.train_mask.numpy(); test_mask = ds.test_mask.numpy()
y_train = node_label[train_mask]; y_test = node_label[test_mask]
node_feat = ds.node_feat

# ---- Cycle-structural reasoning features (deterministic complement of Eq. 9/33) ----
# Per-node closed-walk counts (A^L)_ii count directed closed walks of length L through
# a node; together with in/out degree they form the model's structural cycle signal.
import scipy.sparse as sp
ei_np = ds.edge_index.numpy()
A = sp.csr_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])),
                  shape=(ds.num_nodes, ds.num_nodes))
MAXL = int(os.environ.get("TGRAD_MAXL", "4"))    # longest closed walk in the monitor features
_walks = []; _P = A
for _L in range(2, MAXL + 1):
    _P = _P @ A; _walks.append(_P.diagonal())
struct = np.vstack([np.asarray(A.sum(1)).ravel(), np.asarray(A.sum(0)).ravel()] + _walks).T.astype(np.float32)
print(f"  cycle structural feats: mean (A^3)_ii illicit={struct[node_label==1,3].mean():.3f} "
      f"benign={struct[node_label==0,3].mean():.3f}")
pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], device=DEVICE)
print(f"  test illicit prevalence = {y_test.mean():.3f}")

metrics = {}

# ============================================================================
# 2. BASELINES (attribute-based + structure-aware GCN)
# ============================================================================
print("\n=== SOTA GNN baselines (structure-aware) ===")
from torch_geometric.nn import SAGEConv, GATConv, GINConv, TransformerConv
gnn_ei = to_undirected(ds.edge_index)
gnn_x = ds.node_feat_t
gnn_y = ds.node_label_t
cw = torch.tensor([1.0, pos_weight.item()])

class GNN(nn.Module):
    def __init__(self, kind, d, h=64):
        super().__init__(); self.act = nn.ReLU(); self.drop = nn.Dropout(0.3)
        if kind == "GCN":
            self.c1, self.c2 = GCNConv(d, h), GCNConv(h, 2)
        elif kind == "GraphSAGE":
            self.c1, self.c2 = SAGEConv(d, h), SAGEConv(h, 2)
        elif kind == "GAT":
            self.c1 = GATConv(d, h, heads=4, concat=True, dropout=0.3)
            self.c2 = GATConv(h * 4, 2, heads=1, concat=False, dropout=0.3)
        elif kind == "GIN":
            self.c1 = GINConv(nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h)))
            self.c2 = GINConv(nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2)))
        elif kind == "Graph Transformer":
            self.c1 = TransformerConv(d, h, heads=4, concat=True, dropout=0.3)
            self.c2 = TransformerConv(h * 4, 2, heads=1, concat=False, dropout=0.3)
    def forward(self, x, ei): return self.c2(self.drop(self.act(self.c1(x, ei))), ei)

gnn_probs = {}
def train_gnn(kind, epochs=300, lr=1e-2):
    torch.manual_seed(SEED)
    net = GNN(kind, node_feat.shape[1])
    o = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    for ep in range(epochs):
        net.train(); o.zero_grad()
        loss = nn.functional.cross_entropy(net(gnn_x, gnn_ei)[ds.train_mask], gnn_y[ds.train_mask], weight=cw)
        loss.backward(); o.step()
    net.eval()
    with torch.no_grad():
        s = torch.softmax(net(gnn_x, gnn_ei), 1)[:, 1].numpy()
    gnn_probs[kind] = s
    return evaluate(y_test, s[test_mask], best_threshold(y_train, s[train_mask]))

for kind in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    metrics[kind] = train_gnn(kind)
    print(f"  {kind:<18} AUC={metrics[kind]['ROC-AUC']:.3f} F1={metrics[kind]['F1']:.3f}")

# ============================================================================
# 3. T-GRAD
# ============================================================================
print("\n=== Training T-GRAD ===")
NODE_DIM, TIME_DIM, EDGE_DIM = 128, 32, ds.edge_dim
encoder = TemporalEncoderV2(ds.num_nodes, NODE_DIM, EDGE_DIM, TIME_DIM)
sde = VariancePreservingSDE(ScoreNetworkGNN(NODE_DIM))
guidance = EnergyCycleGuidance(NODE_DIM)
opt = torch.optim.Adam(list(encoder.parameters()) + list(sde.parameters()) +
                       list(guidance.parameters()), lr=1e-3)
node_label_dev = ds.node_label_t
BATCH = 1024
n_train_edges = int((ds.t.numpy() <= ds.train_t_max).sum())
diff_hist, guid_hist = [], []
for ep in range(15):
    encoder.memory.reset_memory(); el = []
    for start in range(0, n_train_edges, BATCH):
        b = ds.get_batch(start, BATCH)
        src, dst, t, msg = b['src'], b['dst'], b['t'], b['msg']
        only = (t <= ds.train_t_max)
        src_z, dst_z = encoder(src, dst, t, msg)
        z = torch.cat([src_z, dst_z], 0); bs = src.shape[0]
        edges = torch.stack([torch.arange(bs), torch.arange(bs, 2 * bs)])
        lbl = torch.cat([node_label_dev[src], node_label_dev[dst]])
        tctx = torch.cat([only, only])
        # labels only from TRAINING entities (an endpoint of a
        # training-window event may be a test entity, whose label must not be used)
        tctx = tctx & torch.cat([ds.train_mask[src], ds.train_mask[dst]])
        licit = (lbl == 0) & tctx
        ld = sde.compute_loss(z[licit], torch.arange(int(licit.sum())).unsqueeze(0).repeat(2, 1)) \
            if licit.sum() > 4 else torch.tensor(0.0)
        lm = (lbl >= 0) & tctx
        lg = guidance.compute_bce_loss(z[lm], lbl[lm], pos_weight=pos_weight) if lm.sum() > 4 else torch.tensor(0.0)
        loss = ld + lg
        if not loss.requires_grad: continue
        opt.zero_grad(); loss.backward(); opt.step(); el.append(loss.item())
        if ep == 14: diff_hist.append(float(ld)); guid_hist.append(float(lg))
    if ep % 3 == 0 or ep == 14:
        print(f"  epoch {ep+1}/15  loss={np.mean(el):.4f}")

# extraction: per-node mean-pooled embedding
encoder.eval(); encoder.memory.reset_memory()
emb_sum = torch.zeros(ds.num_nodes, NODE_DIM); emb_cnt = torch.zeros(ds.num_nodes)
with torch.no_grad():
    for start in range(0, ds.num_edges, BATCH):
        b = ds.get_batch(start, BATCH)
        src, dst, t, msg = b['src'], b['dst'], b['t'], b['msg']
        sz, dz = encoder(src, dst, t, msg)
        emb_sum.index_add_(0, src, sz); emb_cnt.index_add_(0, src, torch.ones_like(t))
        emb_sum.index_add_(0, dst, dz); emb_cnt.index_add_(0, dst, torch.ones_like(t))
node_emb = emb_sum / emb_cnt.clamp(min=1).unsqueeze(1)

with torch.no_grad():
    energy_logit = guidance.fraud_logit(node_emb).numpy()

# guided vs unguided reconstruction anomaly (labeled = all nodes here)
def recon_error(emb, lam, steps=40):
    e = torch.arange(emb.shape[0]).unsqueeze(0).repeat(2, 1)
    tmax = torch.ones(emb.shape[0])
    mean, std = sde.marginal_prob(emb, tmax)
    x = (mean + std[:, None] * torch.randn_like(emb)).requires_grad_(True)
    t_steps = torch.linspace(1.0, 1e-3, steps); dt = 1.0 / steps
    for tv in t_steps:
        tb = torch.ones(x.shape[0]) * tv; beta = sde.beta_t(tv)
        with torch.no_grad():
            score = sde.score_net(x, e, tb)
        if lam > 0:
            score = score + lam * guidance.compute_cycle_gradient(x).clamp(-5, 5)
        drift = -0.5 * beta * x - beta * score
        x = (x - drift * dt + torch.sqrt(torch.tensor(beta)) * torch.sqrt(torch.tensor(dt)) *
             (torch.randn_like(x) if tv > t_steps[-1] else 0)).detach().requires_grad_(True)
    return torch.mean((emb - x) ** 2, 1).detach().numpy()

print("  reconstruction (guided / unguided) ...")
rec_guided = recon_error(node_emb, 10.0)
rec_unguided = recon_error(node_emb, 0.0)

# downstream classifier on augmented representation, selected on held-out val
def std_feat(a):
    return (a - a[train_mask].mean(0)) / (a[train_mask].std(0) + 1e-8)
emb_std = std_feat(node_emb.numpy())
struct_std = std_feat(struct)
X_attr = node_feat
X_full = np.hstack([node_feat, emb_std, struct_std, std_feat(rec_guided.reshape(-1, 1)),
                    std_feat(energy_logit.reshape(-1, 1))])

# held-out validation split inside train for honest config selection
tr_idx = np.where(train_mask)[0]
rng = np.random.default_rng(SEED); rng.shuffle(tr_idx)
val_idx = tr_idx[: len(tr_idx) // 5]; fit_idx = tr_idx[len(tr_idx) // 5:]

class FraudHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_nn(X, fit, epochs=400):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(node_label, dtype=torch.float32)
    h = FraudHead(X.shape[1]); o = torch.optim.Adam(h.parameters(), lr=1e-3, weight_decay=1e-4)
    fm = torch.tensor(fit)
    for _ in range(epochs):
        h.train(); o.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(h(Xt[fm]), yt[fm], pos_weight=pos_weight)
        loss.backward(); o.step()
    h.eval()
    with torch.no_grad():
        return torch.sigmoid(h(Xt)).numpy()

# ablation (neural head, fixed): attributes -> +temporal emb -> +diffusion signals
abl = {}
for tag, X in [("Attributes only", X_attr),
               ("+ Temporal encoder", np.hstack([node_feat, emb_std])),
               ("+ Cycle structural reasoning", np.hstack([node_feat, emb_std, struct_std])),
               ("+ Diffusion guidance (Full)", X_full)]:
    p = train_nn(X, train_mask)
    abl[tag] = evaluate(y_test, p[test_mask], best_threshold(y_train, p[train_mask]))
    print(f"  {tag:<30} AUC={abl[tag]['ROC-AUC']:.3f} F1={abl[tag]['F1']:.3f}")
json.dump(abl, open(os.path.join(ART, "ablation_aml.json"), "w"), indent=2, default=float)

# final T-GRAD = full neural head (retrained on full train)
tgrad_prob = train_nn(X_full, train_mask)
m_tgrad = evaluate(y_test, tgrad_prob[test_mask], best_threshold(y_train, tgrad_prob[train_mask]))
auc_g = roc_auc_score(y_test, rec_guided[test_mask]); auc_u = roc_auc_score(y_test, rec_unguided[test_mask])
m_tgrad["Guidance Gain (AUC)"] = auc_g - auc_u
metrics["T-GRAD"] = m_tgrad
print(f"  {'T-GRAD':<22} AUC={m_tgrad['ROC-AUC']:.3f} F1={m_tgrad['F1']:.3f} "
      f"guidance gain={m_tgrad['Guidance Gain (AUC)']:+.3f}")


# ============================================================================
# EXPERIMENTS REPORTED IN THE PAPER. The block above is kept unchanged so that the
# random stream, and hence every number below, is reproduced exactly.
# ============================================================================
rev = {"maxl": MAXL, "seed": SEED, "published_rows": {}, "baselines_attr": {}, "baselines_matched": {},
       "ablation_factorial": {}, "raw_scores": {}, "timing": {}}
for k, v in abl.items():
    rev["published_rows"][k] = v
for k in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    if k in metrics: rev["baselines_attr"][k] = metrics[k]

# information-matched baselines: every GNN also receives the
# monitor's closed-walk counts and degrees as node features, so any remaining
# gap cannot be attributed to the baselines having been denied that information
gnn_x_matched = torch.tensor(np.hstack([ds.node_feat, struct_std]), dtype=torch.float32)
for kind in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    torch.manual_seed(SEED)
    net = GNN(kind, gnn_x_matched.shape[1])
    o = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
    for ep in range(300):
        net.train(); o.zero_grad()
        loss = nn.functional.cross_entropy(net(gnn_x_matched, gnn_ei)[ds.train_mask],
                                           gnn_y[ds.train_mask], weight=cw)
        loss.backward(); o.step()
    net.eval()
    with torch.no_grad():
        s = torch.softmax(net(gnn_x_matched, gnn_ei), 1)[:, 1].numpy()
    rev["baselines_matched"][kind] = evaluate(y_test, s[test_mask], best_threshold(y_train, s[train_mask]))

# factorial ablation, identical head / optimiser / epochs / threshold
# rule in every row; only the input blocks differ
energy_std = std_feat(energy_logit.reshape(-1, 1))
cells = [("A", X_attr),
         ("A+E", np.hstack([node_feat, emb_std])),
         ("A+M", np.hstack([node_feat, struct_std])),
         ("A+E+M", np.hstack([node_feat, emb_std, struct_std])),
         ("A+E+M+P", np.hstack([node_feat, emb_std, struct_std, energy_std])),
         ("M", struct_std),
         ("E+M", np.hstack([emb_std, struct_std]))]
for tag, X in cells:
    torch.manual_seed(SEED)
    p = train_nn(X, train_mask)
    r = evaluate(y_test, p[test_mask], best_threshold(y_train, p[train_mask]))
    r["n_features"] = int(X.shape[1])
    rev["ablation_factorial"][tag] = r

# raw scores, no classifier and no training on top
w = {2: 2.0**-2, 3: 2.0**-3, 4: 2.0**-4}
mon_raw = w[2]*struct[:, 2] + w[3]*struct[:, 3] + w[4]*struct[:, 4]
rev["raw_scores"]["monitor_weighted_binary"] = evaluate(y_test, mon_raw[test_mask], best_threshold(y_train, mon_raw[train_mask]))
rev["raw_scores"]["monitor_A3_only"] = evaluate(y_test, struct[test_mask, 3], best_threshold(y_train, struct[train_mask, 3]))
rev["raw_scores"]["nodal_posterior_energy_logit"] = evaluate(y_test, energy_logit[test_mask], best_threshold(y_train, energy_logit[train_mask]))

# monitor cost measured on its own, separately from encoding
import time as _t
t0 = _t.perf_counter()
_A = sp.csr_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(ds.num_nodes, ds.num_nodes))
_A2 = _A @ _A; _A3 = _A2 @ _A; _A4 = _A3 @ _A
_ = (_A2.diagonal(), _A3.diagonal(), _A4.diagonal())
rev["timing"]["monitor_sec"] = _t.perf_counter() - t0
rev["timing"]["num_edges"] = int(ei_np.shape[1]); rev["timing"]["num_nodes"] = int(ds.num_nodes)


# ---- degree vs closed walks: the benchmark does NOT degree-match, so separate them ----
deg_std  = struct_std[:, :2]          # out-degree, in-degree
walk_std = struct_std[:, 2:]          # diag(A^2), diag(A^3), diag(A^4)
for tag, X in [("A+D", np.hstack([node_feat, deg_std])),
               ("A+W", np.hstack([node_feat, walk_std])),
               ("A+E+D", np.hstack([node_feat, emb_std, deg_std])),
               ("A+E+W", np.hstack([node_feat, emb_std, walk_std]))]:
    torch.manual_seed(SEED)
    p_ = train_nn(X, train_mask)
    r = evaluate(y_test, p_[test_mask], best_threshold(y_train, p_[train_mask])); r["n_features"] = int(X.shape[1])
    rev["ablation_factorial"][tag] = r

# the exact statistic of Eq. (monitor): L = 2..6, w_L = 2^-L, binary observed adjacency
_P = _A.copy(); mon26 = np.zeros(ds.num_nodes)
for L in range(2, 7):
    _P = _P @ _A
    mon26 += (2.0 ** -L) * _P.diagonal()
deg_tot = struct[:, 0] + struct[:, 1]
for nm, sc in [("monitor_eq_L2to6", mon26), ("total_degree", deg_tot),
               ("in_degree", struct[:, 1]), ("out_degree", struct[:, 0])]:
    rev["raw_scores"][nm] = evaluate(y_test, sc[test_mask], best_threshold(y_train, sc[train_mask]))
# same raw monitor over ALL nodes, no split: the population validate_propositions.py used
from sklearn.metrics import roc_auc_score as _auc
rev["raw_scores"]["monitor_eq_L2to6_ALLNODES_auc"] = float(_auc(node_label, mon26))

# matched GNNs given closed walks only (no degree), to see what walks alone buy them
gnn_x_walk = torch.tensor(np.hstack([ds.node_feat, walk_std]), dtype=torch.float32)
rev["baselines_matched_walks_only"] = {}
for kind in ["GCN", "GraphSAGE", "GAT", "GIN", "Graph Transformer"]:
    torch.manual_seed(SEED)
    net = GNN(kind, gnn_x_walk.shape[1]); o = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
    for ep in range(300):
        net.train(); o.zero_grad()
        loss = nn.functional.cross_entropy(net(gnn_x_walk, gnn_ei)[ds.train_mask], gnn_y[ds.train_mask], weight=cw)
        loss.backward(); o.step()
    net.eval()
    with torch.no_grad(): s_ = torch.softmax(net(gnn_x_walk, gnn_ei), 1)[:, 1].numpy()
    rev["baselines_matched_walks_only"][kind] = evaluate(y_test, s_[test_mask], best_threshold(y_train, s_[train_mask]))

np.savez_compressed(os.path.join(ART, "arrays.npz"), energy_logit=np.asarray(energy_logit).reshape(-1),
                    node_label=node_label, train_mask=train_mask, test_mask=test_mask, monitor_eq=mon26,
                    walks=struct[:, 2:], degree=struct[:, :2])

if os.environ.get("TGRAD_ARTIFACT") == "1":
    # ---- does the LEARNED nodal model actually govern the monitor usefully? ----
    # Thin the stream, re-extract embeddings from the thinned events only (so dropped
    # edges cannot leak into the plausibilities), and score two-hop candidates by the
    # learned plausibility sigma(<z_i, z_j>) of Eq. (softadj) against a constant p and
    # against the same learned values shuffled across candidates (information removed).
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from monitor_utils import two_hop_candidates, monitor as mon_eq
    np.savez_compressed(os.path.join(ART, "embeddings.npz"), node_emb=node_emb.numpy())
    src_all, dst_all = ds.src.numpy(), ds.dst.numpy(); Nn = ds.num_nodes; E_all = len(src_all)
    def extract(keep):
        idx = np.where(keep)[0]; encoder.eval(); encoder.memory.reset_memory()
        es = torch.zeros(Nn, NODE_DIM); ec = torch.zeros(Nn)
        with torch.no_grad():
            for st in range(0, len(idx), BATCH):
                ii = torch.tensor(idx[st:st + BATCH])
                sz, dz = encoder(ds.src[ii], ds.dst[ii], ds.t[ii], ds.msg[ii])
                es.index_add_(0, ds.src[ii], sz); ec.index_add_(0, ds.src[ii], torch.ones(len(ii)))
                es.index_add_(0, ds.dst[ii], dz); ec.index_add_(0, ds.dst[ii], torch.ones(len(ii)))
        z = (es / ec.clamp(min=1).unsqueeze(1)).numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    rng_a = np.random.default_rng(7); lab_all = node_label; rows = []
    for q in (1.0, 0.8, 0.6, 0.4):
        acc = {k: [] for k in ("bin", "const", "learned", "shuffled", "plaus_rank_auc", "mean_plaus")}
        for d in range(5):
            keep = rng_a.random(E_all) < q if q < 1 else np.ones(E_all, bool)
            z = extract(keep)
            so, do_ = src_all[keep], dst_all[keep]
            Ab = sp.coo_matrix((np.ones(len(so)), (so, do_)), shape=(Nn, Nn)).tocsr(); Ab.data[:] = 1
            cr, cc = two_hop_candidates(Ab)
            pl = 1.0 / (1.0 + np.exp(-(z[cr] * z[cc]).sum(1)))
            def soft(w):
                return sp.coo_matrix((np.concatenate([np.ones(len(so)), w]),
                                      (np.concatenate([so, cr]), np.concatenate([do_, cc]))), shape=(Nn, Nn)).tocsr()
            for nm, A_ in [("bin", Ab), ("const", soft(np.full(len(cr), 0.5))),
                           ("learned", soft(pl)), ("shuffled", soft(rng_a.permutation(pl)))]:
                acc[nm].append(roc_auc_score(lab_all, mon_eq(A_)))
            dropped = set(zip(src_all[~keep].tolist(), dst_all[~keep].tolist()))
            is_drop = np.array([(a, b) in dropped for a, b in zip(cr.tolist(), cc.tolist())])
            acc["plaus_rank_auc"].append(roc_auc_score(is_drop, pl) if 0 < is_drop.sum() < len(is_drop) else np.nan)
            acc["mean_plaus"].append(float(pl.mean()))
        row = {"q": q, **{k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in acc.items()}}
        rows.append(row); print("  learned-plausibility", {k: (round(v[0], 3) if isinstance(v, tuple) else v) for k, v in row.items()}, flush=True)
    rev["learned_plausibility"] = rows
json.dump(rev, open(os.path.join(ART, "revision.json"), "w"), indent=2, default=float)
print("\nDONE seed", SEED,
      "| A+E+M", round(rev["ablation_factorial"]["A+E+M"]["ROC-AUC"], 3),
      "| A+M", round(rev["ablation_factorial"]["A+M"]["ROC-AUC"], 3),
      "| best matched GNN", round(max(v["ROC-AUC"] for v in rev["baselines_matched"].values()), 3))
