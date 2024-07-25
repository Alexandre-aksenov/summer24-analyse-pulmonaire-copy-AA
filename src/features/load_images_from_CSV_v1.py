"""
Functions for loading a part of images and masks
in lower resolution
and as black-and-white
from a CSV file.

Called by the script of predictions.
Example test: src/pred_dir_img_mask_10_img_from_CSV.py
"""

import os
import numpy as np
import cv2


def load_images(image_dir: str,
                image_df,
                new_size=(28, 28)):
    """
    Fonction pour charger les images d'un dossier.

    Args:
        image_dir (str): path do data.
        image_df (pd.DataFrame): paths, labels.
        new_size (tuple (int, int)):
            the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).

    Returns:
        tensor NImages x HEIGHT x WIDTH x 2: the images & masks
            resized to HEIGHT x WIDTH.

    Possible improvement:
    refactor out the instructions for reading one
    image or mask.
    """

    num_img = image_df.shape[0]
    image_data = np.zeros((num_img, *new_size, 2))  # tuple unpacking

    # for idx, row in enumerate(image_df):
    for idx, row in image_df.iterrows():
        img_path = os.path.join(image_dir, row['img_folder'], row['Images'])
        try:
            # Read the grayscale image as: HEIGHT x WIDTH x 1
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, new_size)
            img = img / 255.0
            image_data[idx, :, :, 0] = img
        except Exception as e:
            print(f"Erreur de chargement de l'image {img_path} : {e}")

        mask_path = os.path.join(image_dir, row['mask_folder'], row['Masks'])
        try:
            # Read the grayscale mask as: HEIGHT x WIDTH x 1
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, new_size)
            mask = mask / 255.0
            image_data[idx, :, :, 1] = mask
        except Exception as e:
            print(f"Erreur de chargement de l'image {mask_path} : {e}")

    return image_data
