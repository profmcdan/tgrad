"""Closed-walk counts, the cyclic monitor and the two-hop candidate set."""
import numpy as np
import scipy.sparse as sp

MON_LENGTHS = (2, 3, 4, 5, 6)          # the set L of the monitor


def closed_walks(A: sp.csr_matrix, L: int) -> np.ndarray:
    """(A^L)_ii for every node, without forming A^L densely."""
    P = A
    for _ in range(L - 1):
        P = P @ A
    return np.asarray(P.diagonal()).ravel()


def monitor(A: sp.csr_matrix) -> np.ndarray:
    """M_i = sum_L w_L (A^L)_ii with w_L = 2^-L."""
    out = np.zeros(A.shape[0])
    P = A
    for L in range(2, max(MON_LENGTHS) + 1):
        P = P @ A
        if L in MON_LENGTHS:
            out += (2.0 ** -L) * np.asarray(P.diagonal()).ravel()
    return out


def two_hop_candidates(A: sp.csr_matrix):
    """Pairs reachable in two observed hops but not observed directly: the candidate
    set a link-prediction model would score."""
    B = (A > 0).astype(np.int8)
    C = (B @ B).tocsr()
    C = C - C.multiply(B)           # drop pairs that were actually observed
    C.setdiag(0)
    C.eliminate_zeros()
    C = C.tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64)
