"""How the temporal split of the benchmark cuts the planted cycles: the share of cycles with
entities on both sides, and the share of illicit test entities that share a cycle with a
training entity. Replays the generator's random stream to recover each cycle. Run from the
repository root."""
import numpy as np, sys, json
sys.path.append("src")
from tgrad.data.dataset import SyntheticCyclicAML
out = {}
for seed in [42, 0, 1, 2, 3]:
    # replay the generator's RNG stream exactly to recover each cycle's node set
    N, F, NB, NC, lo, hi, span, win = 6000, 165, 18000, 600, 3, 6, 100.0, 1.0
    rng = np.random.default_rng(seed)
    rng.standard_normal((N, F))
    for _ in range(NB):
        a, b = rng.integers(0, N, 2)
        if a == b: continue
        rng.random()
    cycles = []
    for _ in range(NC):
        L = int(rng.integers(lo, hi + 1)); nodes = rng.choice(N, L, replace=False); rng.random()
        cycles.append(nodes)
    ds = SyntheticCyclicAML(num_nodes=N, feat_dim=F, num_benign_edges=NB, num_cycles=NC, seed=seed); ds.load_data()
    tr = ds.train_mask.numpy(); te = ds.test_mask.numpy()
    # sanity: recovered cycles reproduce the dataset's illicit set exactly
    lab = np.zeros(N, int)
    for c in cycles: lab[c] = 1
    assert (lab == ds.node_label_t.numpy()).all(), "RNG replay mismatch"
    straddle = sum(1 for c in cycles if tr[c].any() and te[c].any())
    in_test_only = sum(1 for c in cycles if te[c].all())
    # test illicit nodes whose every cycle also has a node in the training split
    exposed = 0; test_ill = np.where(te & (lab == 1))[0]
    member = {}
    for k, c in enumerate(cycles):
        for v in c: member.setdefault(int(v), []).append(k)
    for v in test_ill:
        if any(tr[cycles[k]].any() for k in member[int(v)]): exposed += 1
    out[seed] = {"cycles": NC, "straddling": straddle, "frac_straddling": straddle / NC,
                 "entirely_in_test": in_test_only,
                 "test_illicit_nodes": int(len(test_ill)),
                 "test_illicit_sharing_a_cycle_with_train": exposed,
                 "frac_test_illicit_exposed": exposed / max(1, len(test_ill))}
    print(seed, out[seed])
json.dump(out, open("results/split_overlap.json", "w"), indent=2)
