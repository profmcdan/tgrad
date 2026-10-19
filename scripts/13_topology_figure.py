"""Force-directed view of a benchmark sub-graph (seed 42): the first 250 events that touch an
illicit entity, with illicit and benign entities coloured. Run from the repository root."""
import os, sys
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append("src")
from tgrad.data.dataset import SyntheticCyclicAML

SEED = 42
ds = SyntheticCyclicAML(seed=SEED); ds.load_data()
ill_set = set(np.where(ds.node_label_t.numpy() == 1)[0].tolist())
ei = ds.edge_index.numpy()
sel = [i for i in range(ei.shape[1]) if ei[0, i] in ill_set or ei[1, i] in ill_set][:250]
G = nx.DiGraph()
for i in sel: G.add_edge(int(ei[0, i]), int(ei[1, i]))
pos = nx.spring_layout(G, k=0.4, iterations=50, seed=SEED)
ncol = ["#d62728" if n in ill_set else "#aec7e8" for n in G.nodes()]
plt.figure(figsize=(9, 7))
nx.draw_networkx_nodes(G, pos, node_size=70, node_color=ncol, alpha=0.9, edgecolors="white", linewidths=0.5)
nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.5, edge_color="#555555", arrowsize=9)
plt.plot([], [], "o", color="#d62728", label="illicit entity (on a planted cycle)")
plt.plot([], [], "o", color="#aec7e8", label="benign entity")
plt.legend(fontsize=11); plt.title("Planted laundering cycles: benchmark sub-graph, seed 42")
plt.axis("off"); plt.tight_layout()
plt.savefig("figures/topology_map.png", dpi=300); plt.close()
