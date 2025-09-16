import numpy as np
from numpy.typing import NDArray

from dreadnode.scorers.image import DistanceMethod


def normalize_for_shape(
    value: float, shape: tuple[int, ...], distance_method: DistanceMethod
) -> float:
    dim_product = np.prod(shape)

    # L0 - no great option here - just return the raw value
    if distance_method == "l0":
        return value

    # L1/Linf - normalize by the product of dimensions
    if distance_method in ["l1", "linf"]:
        return float(value / dim_product)

    # L2 - normalize by the square root of the input dimensions
    if distance_method == "l2":
        return float(value / np.sqrt(dim_product))

    raise ValueError(f"Cannot normalize for unknown distance method '{distance_method}'")


def get_random(
    shape: tuple[int, ...], distance_method: DistanceMethod, *, seed: int | None = None
) -> NDArray[np.float64]:
    generator = np.random.default_rng(seed)  # nosec

    # L1 - Laplace distribution centered at 0 with a scale of 1
    if distance_method == "l1":
        return generator.laplace(size=shape)

    # L2 - Gaussian distribution
    if distance_method == "l2":
        return generator.standard_normal(size=shape)

    # Linf - Uniform distribution between -1 and 1
    if distance_method == "linf":
        return generator.uniform(low=-1, high=1, size=shape)

    raise NotImplementedError(
        f"Cannot generate random noise for '{distance_method}' distance method."
    )
