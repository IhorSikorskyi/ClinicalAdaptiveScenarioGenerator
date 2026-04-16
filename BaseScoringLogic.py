import numpy as np
from scipy.special import softmax

def compute_probabilities(activations: np.ndarray, tau: float = 1.0) -> np.ndarray:
    return softmax(activations / tau)

def shannon_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log2(probs)))

def get_tau(step: int, tau_start: float = 1.2,
            tau_end: float = 0.35, n_steps: int = 10) -> float:
    alpha = (tau_start - tau_end) / n_steps
    return max(tau_end, tau_start - alpha * step)