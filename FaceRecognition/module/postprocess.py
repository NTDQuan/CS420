import math
import numpy as np
from typing import Union, Tuple
import cv2
from PIL import Image

def find_euclidean_distance(
        source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> float:
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def alignment_procedure(
        img: np.ndarray,
        left_eye: tuple,
        right_eye: tuple,
        nose: tuple) -> Tuple[np.ndarray, float, int]:
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1

    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------
    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0],
        # floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173
        # which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    else:
        angle = 0.0  # Dummy value for undefined angle

    # -----------------------

    return img, angle, direction

def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
        angle (float): Angle of rotation in degrees.
        direction (int): Direction of rotation (-1 for clockwise, 1 for counterclockwise).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_facial_area (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) of the rotated facial area.
    """
    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)

def resize_image(
    img: np.ndarray, target_size: Tuple[int, int], min_max_norm: bool = True
) -> np.ndarray:
    """
    Resize an image to expected size of a ml model with adding black pixels.
        Ref: github.com/serengil/deepface/blob/master/deepface/modules/preprocessing.py
    Args:
        img (np.ndarray): pre-loaded image as numpy array
        target_size (tuple): input shape of ml model
        min_max_norm (bool): set this to True if you want to normalize image in [0, 1].
            this is only running when target_size is not none.
            for instance, matplotlib expects inputs in this scale. (default is True)
    Returns:
        img (np.ndarray): resized input image
    """
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    # Put the base image in the middle of the padded image
    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        "constant",
    )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    if min_max_norm is True and img.max() > 1:
        img = (img.astype(np.float32) / 255.0).astype(np.float32)

    return img