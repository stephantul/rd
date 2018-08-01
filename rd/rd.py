"""Representation Distance."""
import numpy as np

from sklearn.metrics import pairwise_distances
from wordkit.features import fourteen, dislex, sixteen
from wordkit.feature_extraction import OneHotCharacterExtractor
from wordkit.transformers import WickelTransformer, \
                                 ConstrainedOpenNGramTransformer, \
                                 WeightedOpenBigramTransformer, \
                                 OpenNGramTransformer, \
                                 LinearTransformer


def rd_features(words,
                n=20,
                feature_name="one hot",
                metric="cosine",
                memory_safe=False):
    """Return the RD of some words fit with some features."""
    X = select_features(words, feature_name)
    return rd(X, n, memory_safe=memory_safe, metric=metric)


def select_features(words, feature_name):
    """Select a feature set and fit the words using that feature set."""
    if feature_name == "one hot":
        f = LinearTransformer(OneHotCharacterExtractor, field=None)
    elif feature_name == "constrained open ngrams":
        f = ConstrainedOpenNGramTransformer(field=None)
    elif feature_name == "weighted open bigrams":
        f = WeightedOpenBigramTransformer((.1, .8, .2), field=None)
    elif feature_name == "open ngrams":
        f = OpenNGramTransformer(field=None)
    elif feature_name == "fourteen":
        f = LinearTransformer(fourteen, field=None)
    elif feature_name == "sixteen":
        f = LinearTransformer(sixteen, field=None)
    elif feature_name == "dislex":
        f = LinearTransformer(dislex, field=None)
    elif feature_name == "wickel":
        f = WickelTransformer(n=3, field=None)
    else:
        raise ValueError("not recognized")

    X = f.fit_transform(words)
    return X


def _cosine_safe(X):
    """Memory safe cosine. Slow."""
    dists = np.zeros((X.shape[0], X.shape[0]))
    X = X / np.linalg.norm(X, axis=1)[:, None]
    for x in range(0, len(X), 100):
        dists[x:x+100] = 1 - X[x:x+100].dot(X.T)
    return dists


def _euclidean_safe(X):
    """Memory safe euclidean distance."""
    dists = []
    for x in range(0, len(X), 100):
        batch = X[x:x+100]
        diff = batch[:, None, :] - X[None, :, :]
        dists.extend(np.linalg.norm(diff, axis=-1))
    return np.array(dists)


def _dist_mtr(X, metric):
    """Separate method because sometimes we want to analyze the matrix."""
    return pairwise_distances(X, metric=metric)


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
    was_int = False
    if isinstance(n, int):
        n = [n]
        was_int = True
    if np.any([x <= 0 for x in n]):
        raise ValueError("n of 0")
    if np.any([x > X.shape[0]-1 for x in n]):
        raise ValueError("Your n was bigger than the number of words - 1.")

    if memory_safe:

        if metric == "cosine":
            dists = _cosine_safe(X)
        elif metric == "euclidean":
            dists = _euclidean_safe(X)
        else:
            raise ValueError("")

    else:
        dists = _dist_mtr(X, metric=metric)

    out = []

    largest_n = max(n)
    d = np.partition(dists, kth=largest_n+1, axis=1)[:, 1:largest_n+1]
    d = np.sort(d, axis=1)
    for x in n:
        out.append(d[:, 1:x+1].mean(1))

    if was_int:
        out = out[0]
    return out
