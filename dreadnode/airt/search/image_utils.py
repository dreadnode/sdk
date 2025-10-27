import typing as t

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from dreadnode.scorers.image import Norm
from dreadnode.util import catch_import_error


def normalize_for_shape(value: float, shape: tuple[int, ...], distance_method: Norm) -> float:
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
    shape: tuple[int, ...], distance_method: Norm, *, seed: int | None = None
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


def clip(array: NDArray[t.Any], min_val: float, max_val: float) -> NDArray[t.Any]:
    original = array.copy()
    clipped = np.clip(array, min_val, max_val)

    total_elements = array.size
    clipped_low = np.sum(original < min_val)
    clipped_high = np.sum(original > max_val)
    clipped_total = clipped_low + clipped_high

    low_violation = np.sum(np.maximum(0, min_val - original))
    high_violation = np.sum(np.maximum(0, original - max_val))
    total_violation = low_violation + high_violation

    actual_change = np.linalg.norm((clipped - original).flatten())

    if clipped_total > 0:
        clip_pct = 100.0 * clipped_total / total_elements
        avg_violation = total_violation / clipped_total if clipped_total > 0 else 0

        logger.debug(
            f"Clipped {clipped_total}/{total_elements} elements "
            f"({clip_pct:.1f}%) | Low: {clipped_low}, High: {clipped_high} | "
            f"Avg violation: {avg_violation:.4f} | L2 change: {actual_change:.4f}"
        )

    return clipped


def create_resizer(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> t.Callable[[NDArray[t.Any]], NDArray[t.Any]]:
    """Creates a function to resize an array using bilinear interpolation."""
    with catch_import_error("scipy"):
        from scipy.ndimage import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
            zoom,
        )

    ndims = len(target_shape)
    factors = [t / s for s, t in zip(source_shape[-ndims:], target_shape[-ndims:], strict=False)]

    def resizer(array: NDArray[t.Any]) -> NDArray[t.Any]:
        array_ndims = len(array.shape)
        full_factors = [1.0] * (array_ndims - ndims) + factors
        return zoom(array, full_factors, order=1, mode="nearest")  # type: ignore[no-any-return]

    return resizer
