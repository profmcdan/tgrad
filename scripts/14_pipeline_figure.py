"""End-to-end diagram of the framework (Figure 1)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "cm"})
fig, ax = plt.subplots(figsize=(7.6, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

C_IN, C_LRN, C_MON, C_OUT = "#eef2f7", "#e6f0e3", "#fbeee0", "#f4f4f4"
def box(x, y, w, h, title, body, fc, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec="#333333", lw=lw))
    nt = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontweight="bold", fontsize=8.2, linespacing=1.15)
    ax.text(x + w / 2, y + h - 1.6 - 3.3 * nt, body, ha="center", va="top", fontsize=7.4, linespacing=1.4)
def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#222222",
                                 connectionstyle=f"arc3,rad={rad}"))
def label(x, y, s, ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=7.0, style="italic", color="#222222")

box(0.5, 21, 13.5, 18, "Transaction\nstream", "events\n$e=(u,v,t,\\mathbf{a})$\nin time order", C_IN)
box(18.5, 38, 20, 19, "Event-driven\nencoder", "time encoding $\\phi(\\Delta t)$\nGRU memory update\nrecency decay\nstate $\\mathbf{z}_i$ per entity", C_LRN)
box(43.5, 38, 21, 19, "Nodal probability\ndistribution model", "class energies $U_0, U_1$\nlogit $f(\\mathbf{z})=\\log\\Lambda(\\mathbf{z})$", C_LRN)
box(70, 48.5, 29.5, 8.5, "Fraud posterior", "$P(y{=}1\\mid\\mathbf{z}_i)$, prior shift (Prop. 1)", C_OUT)
box(70, 35.5, 29.5, 10, "Edge plausibility", "$\\hat p_{ij}=\\sigma(\\cos(\\mathbf{z}_i,\\mathbf{z}_j))$, $(i,j)\\in\\mathcal{C}$", C_LRN)
box(18.5, 4, 20, 19, "Window\naggregation", "$A^{(t)}_{ij}$: number of events\n$i\\to j$ in $[t-\\Delta,t]$\n(event order not kept)", C_IN)
box(43.5, 4, 21, 19, "Plausibility-filled\nadjacency $\\tilde A^{(t)}$", "observed pair: $A^{(t)}_{ij}$\ncandidate pair: $\\hat p_{ij}$\nall other pairs: $0$", C_MON)
box(70, 4, 29.5, 21, "Cyclic monitor", "$M_i(t)=\\sum_{L=2}^{6}2^{-L}\\,((\\tilde A^{(t)})^{L})_{ii}$\nweighted closed walks through $i$\nanalysis: Lemma 1, Prop. 2\noutput: score per entity", C_MON, lw=1.4)

arrow((14.3, 33), (18.2, 45)); label(13.6, 41, "events", ha="right")
arrow((14.3, 27), (18.2, 15)); label(13.6, 19, "events", ha="right")
arrow((38.8, 47.5), (43.2, 47.5)); label(41, 50, "$\\mathbf{z}_i$")
arrow((64.8, 51), (69.7, 52.5))
arrow((64.8, 44), (69.7, 41)); label(67.6, 45.3, "$\\mathbf{z}_i,\\mathbf{z}_j$")
arrow((38.8, 13.5), (43.2, 13.5))
arrow((80, 35.2), (58, 23.3), rad=-0.1); label(73.5, 29.2, "$\\hat p_{ij}$ for unobserved\ncandidate pairs", ha="left")
arrow((64.8, 13.5), (69.7, 13.5))
ax.text(54, 0.4, "Raw monitor in the detection experiments: no candidate pairs ($\\mathcal{C}=\\emptyset$), so $\\tilde A^{(t)}=A^{(t)}$.",
        ha="center", va="bottom", fontsize=7.0, color="#8a4b00")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "pipeline_overview.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); print("wrote", out)
