import typing as t

import numpy as np

from dreadnode.data_types import Image
from dreadnode.metric import Metric
from dreadnode.scorers.base import Scorer

DistanceMethod = t.Literal["l0", "l1", "l2", "linf"]
DistanceMethodName = t.Literal["hamming", "manhattan", "euclidean", "chebyshev"]


def image_distance(
    reference: Image,
    method: DistanceMethod | DistanceMethodName = "l2",
) -> Scorer[Image]:
    """
    Calculates the distance between a candidate image and a reference image
    using a specified metric.

    Args:
        reference: The reference image to compare against.
        method: The distance metric to use. Options are:
            - 'l0' or 'hamming': Counts the number of differing pixels.
            - 'l1' or 'manhattan': Sum of absolute differences (Manhattan distance).
            - 'l2' or 'euclidean': Euclidean distance.
            - 'linf' or 'chebyshev': Maximum absolute difference (Chebyshev distance).
    """

    def evaluate(
        data: Image,
        *,
        reference: Image = reference,
        method: DistanceMethod | DistanceMethodName = method,
    ) -> Metric:
        data_array = data.to_numpy(dtype=np.float32)
        reference_array = reference.to_numpy(dtype=np.float32)
        if data_array.shape != reference_array.shape:
            raise ValueError(
                f"Image shapes do not match: {data_array.shape} vs {reference_array.shape}"
            )

        diff = data_array - reference_array
        distance: float

        if method in ("l2", "euclidean"):
            distance = float(np.linalg.norm(diff.flatten(), ord=2))
        elif method in ("l1", "manhattan"):
            distance = float(np.linalg.norm(diff.flatten(), ord=1))
        elif method in ("linf", "chebyshev"):
            distance = float(np.linalg.norm(diff.flatten(), ord=np.inf))
        elif method in ("l0", "hamming"):
            distance = float(np.linalg.norm(diff.flatten(), ord=0))
        else:
            raise NotImplementedError(f"Distance metric '{method}' not implemented.")

        return Metric(value=distance, attributes={"method": method})

    return Scorer(evaluate, name=f"{method}_distance")
