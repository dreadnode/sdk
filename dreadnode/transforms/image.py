import typing as t

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from dreadnode.data_types import Image
from dreadnode.scorers.image import Norm
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
    alpha: float, *, distance_method: Norm = "l2"
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
        method: Norm = distance_method,
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


def add_text_overlay(
    text: str,
    *,
    position: tuple[int, int] | t.Literal["top", "bottom", "center"] = "bottom",
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 0, 0),  # Red by default
    background_color: tuple[int, int, int, int] | None = (0, 0, 0, 128),  # Semi-transparent black
) -> Transform[Image, Image]:
    """
    Add text overlay to an image using Pillow.

    Args:
        text: The text to add to the image
        position: Either a tuple (x, y) or 'top', 'bottom', 'center'
        font_size: Size of the font
        color: RGB color tuple for text
        background_color: RGBA color tuple for text background (None for no background)

    Returns:
        Transform object that adds text overlay to an Image

    Example:
        >>> transform = add_text_overlay("CONFIDENTIAL", position="top", color=(255, 0, 0))
        >>> modified_image = transform(original_image)
    """

    def transform_func(image: Image) -> Image:
        # Convert to PIL
        pil_img = image.to_pil().convert("RGBA")

        # Create a transparent overlay
        overlay = PILImage.new("RGBA", pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except Exception:  # noqa: BLE001
            try:
                # Try alternative font paths
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:  # noqa: BLE001
                # Fallback to default
                font = t.cast("ImageFont.FreeTypeFont", ImageFont.load_default())

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position
        if isinstance(position, str):
            img_width, img_height = pil_img.size
            if position == "top":
                x = (img_width - text_width) // 2
                y = 10
            elif position == "bottom":
                x = (img_width - text_width) // 2
                y = int(img_height - text_height - 10)
            elif position == "center":
                x = (img_width - text_width) // 2
                y = int(img_height - text_height) // 2
            else:
                x, y = 10, 10
        else:
            x, y = position

        # Draw background rectangle if specified
        if background_color:
            padding = 5
            draw.rectangle(
                [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                fill=background_color,
            )

        # Draw text
        draw.text((x, y), text, font=font, fill=(*color, 255))

        # Composite overlay onto original image
        result = PILImage.alpha_composite(pil_img, overlay)

        if image.mode == "RGB":
            result = result.convert("RGB")

        return Image(result, mode=image.mode, format=image._format)  # noqa: SLF001

    return Transform(transform_func, name=f"add_text_overlay({text})")
