_NUMPY_AVAILABLE = False

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    pass


def cosine_similarity(l1: list[float], l2: list[float]) -> float:
    """Calculates cosine similarity for two lists of floats without external libraries."""
    if len(l1) != len(l2):
        raise ValueError("Vectors must have the same dimension to calculate cosine similarity.")

    if _NUMPY_AVAILABLE:
        v1 = np.array(l1)
        v2 = np.array(l2)
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return float(dot_product / norm_product)

    dot_product = sum(a * b for a, b in zip(l1, l2, strict=True))
    magnitude_l1: float = sum(a**2 for a in l1) ** 0.5
    magnitude_l2: float = sum(b**2 for b in l2) ** 0.5

    # Check for zero-magnitude vectors
    if magnitude_l1 == 0 or magnitude_l2 == 0:
        return 0.0

    return float(dot_product / (magnitude_l1 * magnitude_l2))
