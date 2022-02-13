'''
Author: Maximilian Hageneder
Matrikelnummer: k11942708
'''

from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import dill as pkl
import gzip
from random import randint


def ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    """See assignment sheet for usage description"""
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise NotImplementedError("image_array must be a 2D numpy array")

    border_x_start, border_x_end = border_x
    border_y_start, border_y_end = border_y

    try:  # Check for conversion to int (would raise ValueError anyway but we will write a nice error message)
        border_x_start = int(border_x_start)
        border_x_end = int(border_x_end)
        border_y_start = int(border_y_start)
        border_y_end = int(border_y_end)
    except ValueError as e:
        raise ValueError(f"Could not convert entries in border_x and border_y ({border_x} and {border_y}) to int! "
                         f"Error: {e}")

    if border_x_start < 1 or border_x_end < 1:
        raise ValueError(f"Values of border_x must be greater than 0 but are {border_x_start, border_x_end}")

    if border_y_start < 1 or border_y_end < 1:
        raise ValueError(f"Values of border_y must be greater than 0 but are {border_y_start, border_y_end}")

    remaining_size_x = image_array.shape[0] - (border_x_start + border_x_end)
    remaining_size_y = image_array.shape[1] - (border_y_start + border_y_end)
    if remaining_size_x < 16 or remaining_size_y < 16:
        raise ValueError(f"the size of the remaining image after removing the border must be greater equal (16,16) "
                         f"but was ({remaining_size_x},{remaining_size_y})")

    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[border_x_start:-border_x_end, border_y_start:-border_y_end] = 1

    # Create target_array - don't forget to use .copy(), otherwise target_array and image_array might point to the
    # same array!
    target_array = image_array[known_array == 0].copy()

    # Use image_array as input_array
    image_array[known_array == 0] = 0

    return image_array, known_array, target_array


def cut_and_store(input_path):
    image_files = sorted(glob.glob(os.path.join(input_path, "**", "*.jpg"), recursive=True))
    print(f"Found {len(image_files)} image_files")
    array = []

    with gzip.open(f"ready_for_training.pklz", "w") as fh:
        for i, image_file in tqdm(enumerate(image_files), desc="Preprocessing files", total=len(image_files)):
            picture = edit(image_file)
            picture = np.array(picture)
            picture_crop = picture.copy()
            border_x = randint(5, 9)
            border_y = randint(5, 9)

            pic, known, target = ex4(picture, (border_x, 14 - border_x), (border_y, 14 - border_y))
            array.append((picture_crop, pic, known))
        pkl.dump(dict(array=array), file=fh)


def edit(filename):
    pic_shape = 90
    resize_transforms = transforms.Compose(
        [transforms.Resize(size=pic_shape), transforms.CenterCrop(size=(pic_shape, pic_shape)), ])
    picture = Image.open(filename)
    picture = resize_transforms(picture)
    return picture


cut_and_store("dataset_1")
