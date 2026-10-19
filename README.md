# Cyclic monitoring of transaction streams

Code, benchmark generator and results for the paper

> D. T. Ale, E. O. Ogunti, S. A. Oyetunji, A. A. Ponnle and K. B. Adedeji,
> *A Nodal Probability Distribution Model with Distribution-Governed Cyclic Monitoring for
> Financial Transaction Streams*, Journal of Computing Theories and Applications (under review).

The paper proposes a **cyclic monitor**, a weighted count of the closed walks through each
account in the aggregated graph of a time window, evaluated on an adjacency matrix in which
unobserved candidate pairs can carry a plausibility derived from a nodal probability
distribution model. This repository contains everything needed to regenerate the synthetic
benchmark and every table and figure in the paper.

## Layout

```
src/tgrad/data/dataset.py              benchmark generator (SyntheticCyclicAML) and Elliptic loader
src/tgrad/models/temporal_encoder_v2.py event-driven encoder (memory, time encoding, GRU update)
src/tgrad/models/cycle_guidance.py     energy head: the fraud logit f(z) of the nodal model
src/tgrad/models/diffusion.py          score-matching term trained jointly with the encoder
scripts/                               one script per table or figure (see below)
results/                               the outputs behind every number reported in the paper
figures/                               figures produced by the scripts
```

## Setup

Python 3.13 on CPU. The reported results were produced with torch 2.11.0, torch-geometric
2.7.0, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.2, matplotlib 3.10.9,
seaborn 0.13.2 and networkx 3.6.1.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Elliptic.** The Elliptic Bitcoin dataset is not redistributed here. Download it from Kaggle
(dataset `ellipticco/elliptic-data-set`) and place the three CSV files in
`data/elliptic_bitcoin_dataset/`:

```
data/elliptic_bitcoin_dataset/elliptic_txs_features.csv
data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv
data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv
```

The synthetic benchmark needs no download; it is generated from its seed.

## The benchmark

`SyntheticCyclicAML` (Algorithm 1 in the paper) draws 6,000 entities with 165 standard-normal
attributes, 18,000 benign directed events between uniformly chosen pairs with uniform
timestamps on [0, 100], and 600 planted directed cycles of length 3 to 6 whose hops occur in
order within one per cent of the period. An entity is illicit if it lies on a planted cycle,
and attributes carry no information about the label. Entities are split by the mean time of
their events (first 60% of the period for training).

## Reproducing the paper

Run every command from the repository root. Seeds: `42 0 1 2 3` on the benchmark and
`42 0 1` on Elliptic. Each seed regenerates the benchmark and re-initialises every model.

| Paper content | Script | Output |
|---|---|---|
| Directed-cycle structure of both graphs | `scripts/01_structure.py` | `results/structure.json` |
| How the temporal split cuts the planted cycles | `scripts/02_split_overlap.py` | `results/split_overlap.json` |
| Detection table and component ablation (five seeds) | `scripts/03_benchmark.py`, then `scripts/04_aggregate.py` | `results/benchmark/` |
| Learned plausibilities (seed 42) | `scripts/03_benchmark.py` with `TGRAD_ARTIFACT=1` | `results/benchmark/seed_42_L6/revision.json` |
| Elliptic table and calibration figure | `scripts/05_elliptic.py`, then `scripts/06_elliptic_tables.py` | `results/elliptic/`, `figures/elliptic_posterior_calibration.png` |
| Decay law of Proposition 2 (table and figure) | `scripts/07_decay_law.py` | `results/decay_law.json`, `figures/decay_law.png` |
| Independent against institution-correlated loss | `scripts/08_correlated_loss.py` | `results/correlated_loss.json` |
| Closed walks against simple cycles | `scripts/09_walks_vs_cycles.py` | `results/walks_vs_cycles.json` |
| Collapse of the encoder states; nodal-logit figure | `scripts/10_nodal_collapse.py` | `results/nodal_collapse_seed42.json`, `figures/nodal_logit_train_test.png` |
| Computational cost | `scripts/11_timing.py` | `results/timing.json` |
| Fill of the matrix powers | `scripts/12_fill_growth.py` | `results/fill_growth_200k.json` |
| Benchmark sub-graph figure | `scripts/13_topology_figure.py` | `figures/topology_map.png` |
| Framework diagram | `scripts/14_pipeline_figure.py` | `figures/pipeline_overview.png` |

The benchmark experiments, for the monitor as defined (`TGRAD_MAXL=6`, closed walks of length
2 to 6) and for the truncation check (`TGRAD_MAXL=4`):

```bash
for L in 6 4; do
  for s in 42 0 1 2 3; do
    art=0; [ "$L" = 6 ] && [ "$s" = 42 ] && art=1
    TGRAD_SEED=$s TGRAD_MAXL=$L TGRAD_ARTIFACT=$art python scripts/03_benchmark.py
  done
done
python scripts/04_aggregate.py _L6     # monitor of the paper, L = 2..6
python scripts/04_aggregate.py         # truncation check, L = 2..4
```

The Elliptic experiments:

```bash
for s in 42 0 1; do TGRAD_SEED=$s python scripts/05_elliptic.py; done
python scripts/06_elliptic_tables.py
```

`06_elliptic_tables.py` needs the per-seed `arrays.npz` files written by `05_elliptic.py`. They
contain Elliptic labels and are therefore not included here.

The remaining scripts take no arguments. `13_topology_figure.py` and `07_decay_law.py --plot`
only redraw figures. On a 10-core Apple M5 CPU, one benchmark seed takes about ten minutes and
the analysis scripts take a few minutes each.

Every configuration is fixed in the scripts and listed in Table 7 of the paper. No
hyperparameter was tuned. Labels, feature standardisation and threshold selection use the
training split only; the encoder and energy head are trained on the labels of training
entities only. On the same software versions and CPU, the scripts reproduce the files in
`results/` exactly. Other platforms may differ in the last digits.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The Elliptic dataset is not part of
this repository and remains under the terms of its own release.

## Contact

Daniel T. Ale, Department of Electrical and Electronics Engineering, Federal University of
Technology, Akure, Nigeria.
