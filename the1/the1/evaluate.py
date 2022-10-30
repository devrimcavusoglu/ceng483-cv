import numpy as np

from the1.utils import vnormalize


def kl_divergence(p: np.ndarray, q: np.ndarray):
    """
    Computes KL-Divergence(P, Q).

    Args:
        p: Probabilities of dist P.
        q: Probabilities of dist Q.

    Returns:
        KL-Divergence score.
    """
    # Due to possibility of zero-division or log(0)
    # nan values occur, this function simply ignores them.
    return np.nansum(p * np.log(p/q))


def js_divergence(p: np.ndarray, q: np.ndarray, normalize: bool = True):
    if normalize:
        p = vnormalize(p)
        q = vnormalize(q)
    q_prime = (p + q) / 2
    return (kl_divergence(p, q_prime) + kl_divergence(q, q_prime)) / 2
