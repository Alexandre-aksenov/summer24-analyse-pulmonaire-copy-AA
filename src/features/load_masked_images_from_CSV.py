"""
Functions for loading a part of images and masks
in lower resolution
and as black-and-white
from a CSV file.

Called by the script of predictions.
"""

import os
import numpy as np
import cv2

def load_img_mask(tup_paths: tuple, new_size=(28, 28)):
    """
    Load a single tuple of paths to image, mask. 

    Args:
        tup_paths (tuple): paths to image, mask.
        new_size (tuple (int, int)):d
            the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).

    Returns: tensor 1 x HEIGHT x WIDTH x 3: the masked image resized to HEIGHT x WIDTH in color coding as to 
            match the VGG data entry format.
             
    """
    image_data = np.zeros((1, *new_size, 3))  # tuple unpacking
    
    # Load image
    img_path = tup_paths[0]
    try:
        # Read the grayscale image as: HEIGHT x WIDTH x 1
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, new_size)
    except Exception as e:
        print(f"Erreur de chargement de l'image {img_path} : {e}")
    
    # Load mask
    mask_path = tup_paths[1]
    try:
        # Read the grayscale image as: HEIGHT x WIDTH x 1
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, new_size)
    except Exception as e:
        print(f"Erreur de chargement de l'image {img_path} : {e}")
    masked_img = cv2.bitwise_and(img, mask)/255.0
    # Save the color coded img
    image_data[0, :, :, :] = np.repeat(masked_img.reshape((*new_size,1)),3,-1)
    return image_data


def load_images(image_dir: str,
                image_df,
                new_size=(28, 28)):
    """
    Load a set of masked images from a file

    Args:
        image_dir (str): path do data.
        image_df (pd.DataFrame): paths, labels.
        new_size (tuple (int, int)):
            the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).

    Returns:
        tensor NImages x HEIGHT x WIDTH x 3: the masked images
            resized to HEIGHT x WIDTH in color coding format.

    """

    num_img = image_df.shape[0]
    image_data = np.zeros((num_img, *new_size, 3))  # tuple unpacking

    # for idx, row in enumerate(image_df):
    for idx, row in image_df.iterrows():
        img_path = os.path.join(image_dir, row['img_folder'], row['Images'])
        mask_path = os.path.join(image_dir, row['mask_folder'], row['Masks'])
        image_data[idx, :, :, :] = load_img_mask((img_path, mask_path), new_size=new_size)
    return image_data
