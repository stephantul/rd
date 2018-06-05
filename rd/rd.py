"""Representation Distance."""
import numpy as np

from sklearn.metrics import pairwise_distances


def _cosine_safe(X):
    """Memory safe cosine. Slow."""
    dists = []
    X = X / np.linalg.norm(X, axis=1)[:, None]
    for x in range(0, len(X), 100):
        dists.extend(1 - X[x:x+100].dot(X.T))
    return np.array(dists)


def _euclidean_safe(X):
    """Memory safe euclidean distance."""
    dists = []
    for x in range(0, len(X), 100):
        batch = X[x:x+100]
        diff = batch[:, None, :] - X[None, :, :]
        dists.extend(np.linalg.norm(diff, axis=-1))
    return np.array(dists)


def rd(X, n, memory_safe=False, metric="cosine"):
    """
    Calculate the representation density for values of n.

    n can either be an int or a list of ints

    Parameters
    ----------
    X : np.array
        The representations from which to calculate the density.
    n : int or list of int
        The number of neighbors to take into account.

    Returns
    -------
    densities : np.array
        A vector containing the density for each item.

    """
    if np.any([x <= 0 for x in n]):
        raise ValueError("n of 0")
    if np.any([x > X.shape[0]-1 for x in n]):
        raise ValueError("Your n was bigger than the number of words - 1.")
    if isinstance(n, int):
        n == [n]

    if memory_safe:

        if metric == "cosine":
            dists = _cosine_safe(X)
        elif metric == "euclidean":
            dists = _euclidean_safe(X)
        else:
            raise ValueError("")

    else:
        dists = pairwise_distances(X, metric=metric)

    out = []

    largest_n = max(n)
    d = np.partition(dists, kth=largest_n+1, axis=1)[:, 1:largest_n+1]
    d = np.sort(d, axis=1)
    for x in n:
        out.append(d[:, 1:x+1].mean(1))

    if len(out) == 1:
        out = out[0]
    return out
