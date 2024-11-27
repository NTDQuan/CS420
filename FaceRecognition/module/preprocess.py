import os
import numpy as np
from pathlib import Path
from typing import Union
import cv2

def get_image(img_uri: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img_uri, np.ndarray):
        img = img_uri.copy()
    elif isinstance(img_uri, (str, Path)):
        if isinstance(img_uri, Path):
            img_uri = str(img_uri)

        if not os.path.isfile(img_uri):
            raise ValueError(f"Input image file path ({img_uri}) does not exist")

        img = cv2.imread(img_uri)

    else:
        raise ValueError(
            f"Invalid image input - {img_uri}."
            "Exact paths, pre-loaded numpy arrays, base64 encoded "
            "strings and urls are welcome."
        )

    if len(img.shape) != 3 or np.prod(img.shape) != 3:
        raise ValueError("Input image needs to have 3 channels at must not be empty")

    return img


