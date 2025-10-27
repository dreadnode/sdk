import typing as t

import numpy as np

from dreadnode.data_types import Image
from dreadnode.metric import Metric
from dreadnode.scorers.base import Scorer

Norm = t.Literal["l0", "l1", "l2", "linf"]


def image_distance(
    reference: Image,
    norm: Norm = "l2",
    *,
    normalize: bool = False,
) -> Scorer[Image]:
    """
    Calculates the distance between a candidate image and a reference image
    using a specified metric.

    Optionally you can normalize the distance to a [0, 1] range based on
    the shape of the image (assumes the images are in [0, 1] range).

    Args:
        reference: The reference image to compare against.
        norm: The distance metric to use. Options are:
            - 'l0' or 'hamming': Counts the number of differing pixels.
            - 'l1' or 'manhattan': Sum of absolute differences (Manhattan distance).
            - 'l2' or 'euclidean': Euclidean distance.
            - 'linf' or 'chebyshev': Maximum absolute difference (Chebyshev distance).
        normalize: If True, normalizes the distance to a [0, 1] range.
    """

    def evaluate(
        data: Image,
        *,
        reference: Image = reference,
        method: Norm = norm,
        normalize: bool = normalize,
    ) -> Metric:
        if not isinstance(data, Image):
            raise TypeError(f"Expected data to be an Image, got {type(data)}")
        if not isinstance(reference, Image):
            raise TypeError(f"Expected reference to be an Image, got {type(reference)}")

        data_array = data.to_numpy()
        reference_array = reference.to_numpy()
        if data_array.shape != reference_array.shape:
            raise ValueError(
                f"Image shapes do not match: {data_array.shape} vs {reference_array.shape}"
            )

        diff = data_array - reference_array
        distance: float

        if method == "l0":
            distance = float(np.linalg.norm(diff.flatten(), ord=0))
            if normalize:
                # Max L0 distance is the total number of elements
                max_dist = float(data_array.size)
                distance /= max_dist
        elif method == "l1":
            distance = float(np.linalg.norm(diff.flatten(), ord=1))
            if normalize:
                # Max L1 distance is N * 1.0 = N
                max_dist = float(data_array.size)
                distance /= max_dist
        elif method == "l2":
            distance = float(np.linalg.norm(diff.flatten(), ord=2))
            if normalize:
                # Max L2 distance is sqrt(N * (1.0^2)) = sqrt(N)
                # where N is the number of elements (pixels * channels)
                max_dist = np.sqrt(data_array.size)
                distance /= max_dist
        elif method == "linf":
            distance = float(np.linalg.norm(diff.flatten(), ord=np.inf))
            if normalize:
                # Max Linf distance is the max difference for a single element, which is 1.0
                max_dist = 1.0
                distance /= max_dist
        else:
            raise NotImplementedError(f"Distance metric '{method}' not implemented.")

        return Metric(value=distance, attributes={"method": method, "normalized": normalize})

    return Scorer(evaluate, name=f"{norm}_distance")
