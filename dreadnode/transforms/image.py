import numpy as np

from dreadnode.data_types import Image
from dreadnode.transforms.base import Transform


def add_gaussian_noise(
    std_dev: float = 0.05, *, seed: int | None = None
) -> Transform[Image, Image]:
    """Adds Gaussian noise to an image."""

    random = np.random.RandomState(seed)  # nosec

    def transform(image: Image) -> Image:
        image_array = image.to_numpy(dtype=np.float32)
        noise = random.normal(0, std_dev, image_array.shape)
        return Image(np.clip(image_array + noise, 0, 1))

    return Transform(transform, name="add_gaussian_noise")


def add_laplace_noise(scale: float = 0.05, *, seed: int | None = None) -> Transform[Image, Image]:
    """Adds Laplace noise to an image."""

    random = np.random.RandomState(seed)  # nosec

    def transform(image: Image) -> Image:
        image_array = image.to_numpy(dtype=np.float32)
        noise = random.laplace(0, scale, image_array.shape)
        return Image(np.clip(image_array + noise, 0, 1))

    return Transform(transform, name="add_laplace_noise")


def add_uniform_noise(
    low: float = -0.05, high: float = 0.05, *, seed: int | None = None
) -> Transform[Image, Image]:
    """Adds Uniform noise to an image."""

    random = np.random.RandomState(seed)  # nosec

    def transform(image: Image, *, low: float = low, high: float = high) -> Image:
        image_array = image.to_numpy(dtype=np.float32)
        noise = random.uniform(low, high, image_array.shape)  # nosec
        return Image(np.clip(image_array + noise, 0, 1))

    return Transform(transform, name="add_uniform_noise")


def shift_pixel_values(max_delta: int = 5, *, seed: int | None = None) -> Transform[Image, Image]:
    """Randomly shifts pixel values by a small integer amount."""

    random = np.random.RandomState(seed)  # nosec

    def transform(image: Image, *, max_delta: int = max_delta) -> Image:
        image_array = image.to_numpy()
        delta = random.randint(-max_delta, max_delta + 1, image_array.shape)  # nosec
        return Image(np.clip(image_array + delta, 0, 255).astype(np.uint8))

    return Transform(transform, name="shift_pixel_values")


def interpolate(alpha: float) -> Transform[tuple[Image, Image], Image]:
    """
    Creates a transform that performs linear interpolation between two images.

    The returned image is calculated as: `(1 - alpha) * start + alpha * end`.

    Args:
        alpha: The interpolation factor. 0.0 returns the start image,
               1.0 returns the end image. 0.5 is the midpoint.

    Returns:
        A Transform that takes a tuple of (start_image, end_image) and
        returns the interpolated image.
    """

    def transform(images: tuple[Image, Image], *, alpha: float = alpha) -> Image:
        start_image, end_image = images

        start_np = start_image.to_numpy(dtype=np.float32)
        end_np = end_image.to_numpy(dtype=np.float32)

        if start_np.shape != end_np.shape:
            raise ValueError(
                f"Cannot interpolate between images with different shapes: "
                f"{start_np.shape} vs {end_np.shape}"
            )

        interpolated_np = (1.0 - alpha) * start_np + alpha * end_np
        return Image(interpolated_np)

    # The name helps with logging and debugging
    return Transform(transform, name=f"interpolate(alpha={alpha:.2f})")
