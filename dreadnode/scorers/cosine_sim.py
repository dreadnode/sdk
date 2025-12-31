import numpy as np


def cosine_similarity(l1: list[float], l2: list[float]) -> float:
    """Calculates cosine similarity for two lists of floats without external libraries."""
    if len(l1) != len(l2):
        raise ValueError("Vectors must have the same dimension to calculate cosine similarity.")

    v1 = np.array(l1)
    v2 = np.array(l2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
    return float(dot_product / norm_product)
