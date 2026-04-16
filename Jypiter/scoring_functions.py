import numpy as np
from scipy.special import softmax


def compute_probabilities(activations, tau=1.0):
    return softmax(activations / tau)


def shannon_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def get_tau(step, tau_start=1.2, tau_end=0.35, n_steps=10):
    """Temperature annealing schedule."""
    alpha = (tau_start - tau_end) / n_steps
    return max(tau_end, tau_start - alpha * step)
