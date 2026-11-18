import numpy as np
from PIL import Image, ImageEnhance


class PrepareViTImage:
    """Boost dull colours a little while leaving strong colours mostly alone."""

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        highest = rgb.max(axis=2, keepdims=True)
        lowest = rgb.min(axis=2, keepdims=True)
        saturation = (highest - lowest) / np.maximum(highest, 1e-6)
        grey = (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2, keepdims=True)
        # Up to 8% extra colour for muted pixels; already vivid pixels get less.
        vibrant = grey + (rgb - grey) * (1.0 + 0.08 * (1.0 - saturation))
        result = Image.fromarray(np.rint(np.clip(vibrant, 0, 1) * 255).astype(np.uint8))
        return ImageEnhance.Contrast(result).enhance(1.05)
