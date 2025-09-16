import numpy as np

from dreadnode.data_types import Image
from dreadnode.scorers.image import DistanceMethod
from dreadnode.transforms.base import Transform


def add_gaussian_noise(*, scale: float = 1, seed: int | None = None) -> Transform[Image, Image]:
    """Adds Gaussian noise to an image."""

    random = np.random.default_rng(seed)  # nosec

    def transform(image: Image, *, scale: float = scale) -> Image:
        image_array = image.to_numpy()
        noise = random.normal(scale=scale, size=image_array.shape)
        return Image(np.clip(image_array + noise, 0, 1))

    return Transform(transform, name="add_gaussian_noise")


def add_laplace_noise(*, scale: float = 1, seed: int | None = None) -> Transform[Image, Image]:
    """Adds Laplace noise to an image."""

    random = np.random.default_rng(seed)  # nosec

    def transform(image: Image, *, scale: float = scale) -> Image:
        image_array = image.to_numpy()
        noise = random.laplace(scale=scale, size=image_array.shape)
        return Image(np.clip(image_array + noise, 0, 1))

    return Transform(transform, name="add_laplace_noise")


def add_uniform_noise(
    *, low: float = -1, high: float = 1, seed: int | None = None
) -> Transform[Image, Image]:
    """Adds Uniform noise to an image."""

    random = np.random.default_rng(seed)  # nosec

    def transform(image: Image, *, low: float = low, high: float = high) -> Image:
        image_array = image.to_numpy()
        noise = random.uniform(low=low, high=high, size=image_array.shape)  # nosec
        return Image(np.clip(image_array + noise, 0, 1))

    return Transform(transform, name="add_uniform_noise")


def shift_pixel_values(max_delta: int = 5, *, seed: int | None = None) -> Transform[Image, Image]:
    """Randomly shifts pixel values by a small integer amount."""

    random = np.random.default_rng(seed)  # nosec

    def transform(image: Image, *, max_delta: int = max_delta) -> Image:
        image_array = image.to_numpy(dtype=np.int8)
        delta = random.integers(low=-max_delta, high=max_delta + 1, size=image_array.shape)  # nosec
        return Image(image_array + delta)

    return Transform(transform, name="shift_pixel_values")


def interpolate_images(
    alpha: float, *, distance_method: DistanceMethod = "l2"
) -> Transform[tuple[Image, Image], Image]:
    """
    Creates a transform that performs linear interpolation between two images.

    The returned image is calculated as: `(1 - alpha) * start + alpha * end`.

    Args:
        alpha: The interpolation factor. 0.0 returns the start image,
               1.0 returns the end image. 0.5 is the midpoint.
        distance_method: The distance method being used - for optimizing interpolation.

    Returns:
        A Transform that takes a tuple of (start_image, end_image) and
        returns the interpolated image.
    """

    def transform(
        images: tuple[Image, Image],
        *,
        alpha: float = alpha,
        method: DistanceMethod = distance_method,
    ) -> Image:
        start_image, end_image = images

        start_np = start_image.to_numpy()
        end_np = end_image.to_numpy()

        if start_np.shape != end_np.shape:
            raise ValueError(
                f"Cannot interpolate between images with different shapes: "
                f"{start_np.shape} vs {end_np.shape}"
            )

        # Linf - we do a simple clip to ensure we don't exceed the max difference
        if method == "linf":
            interpolated_np = np.clip(end_np, start_np - alpha, start_np + alpha)

        # L0/L1/L2, we do standard linear interpolation
        elif method in ("l0", "l1", "l2"):
            interpolated_np = (1.0 - alpha) * start_np + alpha * end_np

        return Image(interpolated_np)

    return Transform(transform, name="interpolate")
