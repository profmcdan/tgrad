"""Time the encoding pass and the cyclic monitor SEPARATELY, on
AMLSim-style replicas with the scalability study's generator settings, on an idle
machine, and record the hardware. Weights are irrelevant to cost, so the encoder
is timed untrained; the monitor is the statistic of Eq. (monitor), L = 2..6."""
import os, sys, time, json, platform, subprocess, numpy as np, torch, scipy.sparse as sp
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from tgrad.data.dataset import SyntheticCyclicAML
from tgrad.models.temporal_encoder_v2 import TemporalEncoderV2
from tgrad.models.cycle_guidance import EnergyCycleGuidance
torch.manual_seed(42); torch.set_num_threads(int(os.environ.get("TGRAD_THREADS", "8")))
MEAN_DEG, CYCLE_EDGE_FRAC, AVG_CYCLE_LEN = 3.45, 0.13, 4.5
NODE_DIM, TIME_DIM, BATCH, FEAT = 128, 32, 2048, 64
SIZES = [int(x) for x in os.environ.get("TGRAD_SIZES", "200000,400000,600000,800000,1000000,1200000,2000000").split(",")]
def sysinfo():
    g = lambda k: subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
    return {"cpu": g("machdep.cpu.brand_string"), "cores": g("hw.ncpu"), "ram_gb": round(int(g("hw.memsize")) / 2**30, 1),
            "os": "macOS " + platform.mac_ver()[0], "python": platform.python_version(),
            "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
rows = []
for E in SIZES:
    n = int(round(E / MEAN_DEG)); nc = int(round(CYCLE_EDGE_FRAC * E / AVG_CYCLE_LEN)); nb = int(round(E - nc * AVG_CYCLE_LEN))
    ds = SyntheticCyclicAML(num_nodes=n, feat_dim=FEAT, num_benign_edges=nb, num_cycles=nc, cycle_len_range=(3, 6), time_span=100.0, seed=42)
    ds.load_data()
    enc = TemporalEncoderV2(ds.num_nodes, NODE_DIM, ds.edge_dim, TIME_DIM); enc.eval(); enc.memory.reset_memory()
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, ds.num_edges, BATCH):
            b = ds.get_batch(s, BATCH); enc(b["src"], b["dst"], b["t"], b["msg"])
    t_enc = time.perf_counter() - t0
    # nodal model: one energy-head forward pass per entity (the posterior of Eq. posterior)
    head = EnergyCycleGuidance(NODE_DIM); head.eval(); zz = torch.randn(ds.num_nodes, NODE_DIM)
    t0 = time.perf_counter()
    with torch.no_grad(): head.fraud_logit(zz)
    t_head = time.perf_counter() - t0
    ei = ds.edge_index.numpy()
    t0 = time.perf_counter()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(ds.num_nodes, ds.num_nodes)); A.data[:] = 1
    # exact and memory-bounded: diag(A^L) = rowsum(A^a * (A^b)^T), a + b = L, so only
    # A^2 and A^3 are formed (fill ~ d^3) instead of A^6 (fill ~ d^6, infeasible at 2M edges)
    A2 = A @ A; A3 = A2 @ A; pw = {1: A, 2: A2, 3: A3}; M = np.zeros(ds.num_nodes)
    for L, (a, b) in {2: (1, 1), 3: (1, 2), 4: (2, 2), 5: (2, 3), 6: (3, 3)}.items():
        M += (2.0 ** -L) * np.asarray(pw[a].multiply(pw[b].T).sum(1)).ravel()
    t_mon = time.perf_counter() - t0; P = A3
    r = {"edges": int(ds.num_edges), "nodes": int(ds.num_nodes), "encoder_sec": t_enc, "encoder_events_per_sec": ds.num_edges / t_enc,
         "energy_head_sec": t_head, "monitor_sec": t_mon, "monitor_events_per_sec": ds.num_edges / t_mon, "nnz_A3": int(P.nnz)}
    rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump({"hardware": sysinfo(), "rows": rows}, open("results/timing.json", "w"), indent=2)
print(sysinfo())
