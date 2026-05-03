from pathlib import Path
from typing import Optional, Tuple, Union, List

import cv2
import math
import numpy as np
from PIL import Image, ImageOps

InputType = Union[str, np.ndarray, bytes, Path]


class PPPreProcess:
    def __init__(self, img_size: Tuple[int, int]):
        self.uni_mer_net_img_decode = UniMERNetImgDecode(img_size)
        self.uni_mer_net_test_transform = UniMERNetTestTransform()
        self.latex_image_format = LatexImageFormat()

    def __call__(self, imgs: List[np.ndarray] = None) -> List[np.ndarray]:
        batch_imgs = self.uni_mer_net_img_decode(imgs=imgs)
        batch_imgs = self.uni_mer_net_test_transform(imgs=batch_imgs)
        batch_imgs = self.latex_image_format(imgs=batch_imgs)
        return batch_imgs

class UniMERNetImgDecode(object):
    """Class for decoding images for UniMERNet, including cropping margins, resizing, and padding."""

    def __init__(
        self, input_size: Tuple[int, int], random_padding: bool = False, **kwargs
    ) -> None:
        """Initializes the UniMERNetImgDecode class with input size and random padding options.

        Args:
            input_size (tuple): The desired input size for the images (height, width).
            random_padding (bool): Whether to use random padding for resizing.
            **kwargs: Additional keyword arguments."""
        self.input_size = input_size
        self.random_padding = random_padding

    def crop_margin(self, img: Image.Image) -> Image.Image:
        """Crops the margin of the image based on grayscale thresholding.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The cropped image."""
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def get_dimensions(self, img: Union[Image.Image, np.ndarray]) -> List[int]:
        """Gets the dimensions of the image.

        Args:
            img (PIL.Image.Image or numpy.ndarray): The input image.

        Returns:
            list: A list containing the number of channels, height, and width."""
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]

    def _compute_resized_output_size(
        self,
        image_size: Tuple[int, int],
        size: Union[int, Tuple[int, int]],
        max_size: Optional[int] = None,
    ) -> List[int]:
        """Computes the resized output size of the image.

        Args:
            image_size (tuple): The original size of the image (height, width).
            size (int or tuple): The desired size for the smallest edge or both height and width.
            max_size (int, optional): The maximum allowed size for the longer edge.

        Returns:
            list: A list containing the new height and width."""
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short
            )

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def resize(
        self, img: Image.Image, size: Union[int, Tuple[int, int]]
    ) -> Image.Image:
        """Resizes the image to the specified size.

        Args:
            img (PIL.Image.Image): The input image.
            size (int or tuple): The desired size for the smallest edge or both height and width.

        Returns:
            PIL.Image.Image: The resized image."""
        _, image_height, image_width = self.get_dimensions(img)
        if isinstance(size, int):
            size = [size]
        max_size = None
        output_size = self._compute_resized_output_size(
            (image_height, image_width), size, max_size
        )
        img = img.resize(tuple(output_size[::-1]), resample=2)
        return img

    def img_decode(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Decodes the image by cropping margins, resizing, and adding padding.

        Args:
            img (numpy.ndarray): The input image array.

        Returns:
            numpy.ndarray: The decoded image array."""
        try:
            img = self.crop_margin(Image.fromarray(img).convert("RGB"))
        except OSError:
            return
        if img.height == 0 or img.width == 0:
            return
        img = self.resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if self.random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return np.array(ImageOps.expand(img, padding))

    def __call__(self, imgs: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Calls the img_decode method on a list of images.

        Args:
            imgs (list of numpy.ndarray): The list of input image arrays.

        Returns:
            list of numpy.ndarray: The list of decoded image arrays."""
        return [self.img_decode(img) for img in imgs]


class UniMERNetTestTransform:
    """
    A class for transforming images according to UniMERNet test specifications.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the UniMERNetTestTransform class.
        """
        super().__init__()
        self.num_output_channels = 3

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Transforms a single image for UniMERNet testing.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The transformed image.
        """
        mean = [0.7931, 0.7931, 0.7931]
        std = [0.1738, 0.1738, 0.1738]
        scale = float(1 / 255.0)
        shape = (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")
        img = (img.astype("float32") * scale - mean) / std
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squeezed = np.squeeze(grayscale_image)
        img = cv2.merge([squeezed] * self.num_output_channels)
        return img

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies the transform to a list of images.

        Args:
            imgs (list of numpy.ndarray): The list of input images.

        Returns:
            list of numpy.ndarray: The list of transformed images.
        """
        return [self.transform(img) for img in imgs]

class LatexImageFormat:
    """Class for formatting images to a specific format suitable for LaTeX."""

    def __init__(self, **kwargs) -> None:
        """Initializes the LatexImageFormat class with optional keyword arguments."""
        super().__init__()

    def format(self, img: np.ndarray) -> np.ndarray:
        """Formats a single image to the LaTeX-compatible format.

        Args:
            img (numpy.ndarray): The input image as a numpy array.

        Returns:
            numpy.ndarray: The formatted image as a numpy array with an added dimension for color.
        """
        im_h, im_w = img.shape[:2]
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = img[:, :, 0]
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img_expanded = img[:, :, np.newaxis].transpose(2, 0, 1)
        return img_expanded[np.newaxis, :]

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Applies the format method to a list of images.

        Args:
            imgs (list of numpy.ndarray): A list of input images as numpy arrays.

        Returns:
            list of numpy.ndarray: A list of formatted images as numpy arrays.
        """
        return [self.format(img) for img in imgs]