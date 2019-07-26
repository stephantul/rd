"""Representation Distance."""
import numpy as np

from tqdm import tqdm
from sklearn.metrics import pairwise_distances


def _cosine_safe(X, Y):
    """Memory safe cosine. Slow."""
    dists = np.zeros((X.shape[0], Y.shape[0]))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    if X is not Y:
        Y /= np.linalg.norm(Y, axis=1)[:, None]
    X /= np.linalg.norm(X, axis=1)[:, None]
    for x in tqdm(range(0, len(X), 100)):
        dists[x:x+100] = 1 - X[x:x+100].dot(Y.T)
    return dists


def _euclidean_safe(X, Y):
    """Memory safe euclidean distance."""
    dists = []
    for x in tqdm(range(0, len(X), 100)):
        batch = X[x:x+100]
        diff = batch[:, None, :] - Y[None, :, :]
        dists.extend(np.linalg.norm(diff, axis=-1))
    return np.array(dists)


def dist_mtr(X, Y, metric):
    """Separate method because sometimes we want to analyze the matrix."""
    if X is Y:
        return pairwise_distances(X, metric=metric)
    return pairwise_distances(X, Y, metric=metric)


def rd(X, Y=None, n=20, memory_safe=False, metric="cosine"):
    """
    Calculate the representation density for values of n.

    n can either be an int or a list of ints

    If only X is given, the density will be computed based on X * X
    if Y is also given, the density will be computed based on X * Y

    Parameters
    ----------
    X : np.array
        The representations for which to calculate the density
    Y : np.array
        The reference representations to use
    n : int or list of int
        The number of neighbors to take into account.

    Returns
    -------
    densities : np.array
        A vector containing the density for each item.

    """
    if isinstance(X, list):
        X = np.asarray(X)
    was_int = False
    if Y is None:
        Y = X
    if isinstance(n, int):
        n = [n]
        was_int = True
    if np.any([x <= 0 for x in n]):
        raise ValueError("n of 0")
    if np.any([x > X.shape[0]-1 for x in n]):
        raise ValueError("Your n was bigger than the number of words - 1.")

    if memory_safe:

        if metric == "cosine":
            dists = _cosine_safe(X, Y)
        elif metric == "euclidean":
            dists = _euclidean_safe(X, Y)
        else:
            raise ValueError("Metric not available in safe mode.")

    else:
        dists = dist_mtr(X, Y, metric=metric)

    out = []

    largest_n = max(n)
    d = np.partition(dists, kth=largest_n+1, axis=1)[:, :largest_n+1]
    d = np.sort(d, axis=1)
    for x in n:
        out.append(d[:, :x+1].sum(1))

    if was_int:
        out = out[0]
    return out
