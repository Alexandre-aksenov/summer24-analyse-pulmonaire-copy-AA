"""
Functions for loading a part of images and masks
in lower resolution
and as black-and-white
from 2 directories.

The file names from the CSV files
should receive the ending .png, then excluded.

This DF should be transferred down the function
'list_img' and treated there
using file_name[:-4] in  excluded.to_numpy()

Will be used in
train_val_test_split_save_v1_2.ipynb
TOWRITE
"""

import os
import numpy as np
import cv2
# import pandas as pd


def to_inlude(fname: str, ext='.png', excluded=None):
    res1 = fname.endswith(ext)
    if res1:
        res = excluded is None or (not fname[:-4] in excluded.to_numpy())
    else:
        res = False
    return res

#    return fname.endswith(ext) and not fname[:-4] in excluded.to_numpy()


def list_img(image_dir: str, ext='.png', excluded=None):
    """
    List of files with a specific extension in a folder
    in the alphabetical order.

    Useful for extracting the number of files.
    Serves also as the first step of the function 'load_images'.

    Args:
        image_dir (str): path to folder
        ext (str, optional): sought extension. Defaults to '.png'.
        excluded (pd object with file names or None): the file names to exclude
            Defaults to None.

    Returns:
        list of str: the list of image files in the order of os.listdir.
    """

    files = sorted(os.listdir(image_dir))
    # alphabetical sorting: https://datascientest.com/python-string-tout-savoir

    lst_images = [file_name for file_name in files if to_inlude(file_name, ext='.png', excluded=excluded)]
    return lst_images


def load_images(image_dir,
                label,
                new_size=(28, 28),
                excluded=None,
                limit=None,
                random_state=None):
    """
    Fonction pour charger les images d'un dossier.

    Args:
        image_dir (str): path to folder.
        label (str or int): the label to associate to the images.
        new_size (tuple (int, int)):
            the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).
        excluded(pd object with file names or None): the file names to exclude
            Defaults to None.
        limit (int, optional): the max number of files to load.
            Defaults to None (load all PNG images).
        random_state (None or int): controls the random generator.

    Called by the function: load_data.

    Returns:
        tensor NImages x HEIGHT x WIDTH x 1: the images
            resized to HEIGHT x WIDTH;
        1d array (NImages,) : the label expanded to array.
        files: the list of file names.

    Possible improvements.
    1. Speed up the function by preallocating the tensor first,
    then writing into it.

    4. Return the  filenames of the extracted images as a 3rd output.
    """
    image_data = []
    label_data = []
    rng = np.random.default_rng(seed=random_state)

    all_files = list_img(image_dir, excluded=None)

    assert(limit is None or limit <= len(all_files))
    # otherwize, the next instruction will fail.

    if limit:
        files = rng.choice(all_files, size=limit, replace=False)
    else:
        files = all_files

    for file_name in files:
        img_path = os.path.join(image_dir, file_name)
        try:
            # Read the grayscale image as: HEIGHT x WIDTH x 1

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, new_size)
            img = np.expand_dims(img, axis=-1)

            img = img / 255.0
            image_data.append(img)
            label_data.append(label)
        except Exception as e:
            print(f"Erreur de chargement de l'image {file_name} : {e}")

    return np.array(image_data), np.array(label_data), files


def load_data(image_dirs,
              labels,
              new_size=(28, 28),
              excluded=None,
              limits=None,
              random_state=None):
    """
    Load data from several folders

    Args:
        image_dirs (list of 'str'): the paths to folders.
        labels (list of 'str' of the same length):
            labels to associate to each directory.
        new_size (tuple (int, int)):
            the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).
        excluded(pd object with file names or None): the file names to exclude
            Defaults to None.
        limits (list if int of None, optional): the number of files to load.
            Defaults to None.
        random_state (None or int): controls the random generator.

    Returns:
        tensor sum_{folders}(NImages) x HEIGHT x WIDTH :
            the images resized to HEIGHT x WIDTH;
        1d array (sum_{folders}(NImages),) : the labels expanded to array.
        all_file_names : list of lists of file names,
            each sublist corresponds to a "condition" (COVID, Pneumonia, Normal).
        ->
        all_file_names : list of file names.
    """
    all_images = []
    all_labels = []
    all_file_names = []

    limits2 = [None] * len(image_dirs) if limits is None else limits

    for i, image_dir in enumerate(image_dirs):
        images, label_data, fnames = load_images(image_dir,
                                                 labels[i],
                                                 excluded=excluded,
                                                 limit=limits2[i],
                                                 new_size=new_size,
                                                 random_state=random_state)
        all_images.append(images)
        all_labels.append(label_data)
        # all_file_names.append(fnames)
        all_file_names.extend(fnames)
    return np.concatenate(all_images), np.concatenate(all_labels), all_file_names


def load_data_masks(image_dirs,
                    labels,
                    new_size=(28, 28),
                    excluded=None,
                    limits=None,
                    random_state=None):
    """
    Load data from several folders grouped as a list of lists.

    Args:
        image_dirs (list of lists of 'str'): the paths to folders.
            The sublists should be of same length.

            The function attempts to match all folders
            at the same position in different lists,
            so that the folders should contain the same of image files,
            and ideally these files should have same names.

        labels (list of 'str' of the same length as sublists above):
            labels to associate to each directory.
        new_size (tuple (int, int)):
            the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).
        excluded(pd object with file names or None): the file names to exclude
            Defaults to None.
        limits (list if int of None, optional): the number of files to load.
            Defaults to None.
        random_state (None or int): controls the random generator.

    Returns:
        tensor sum_{folders}(NImages) x HEIGHT x WIDTH x num_modalities:
            the images resized to HEIGHT x WIDTH;
        1d array (sum_{folders}(NImages),) : the labels expanded to array.
        all_file_mames: list of lists of lists of file names,
            the sublists correspond to images, masks,
            each sublist corresponds to a "condition"
            (COVID, Pneumonia, Normal).
    """
    all_images = []
    all_file_names = []

    for idx, modality in enumerate(image_dirs):
        if (idx == 0):
            data_img, label_data, fnames = load_data(modality,
                        labels,
                        new_size=new_size,
                        excluded=excluded,
                        limits=limits,
                        random_state=random_state)
        else:
            data_img, __, fnames = load_data(modality,
                        labels,
                        new_size=new_size,
                        excluded=excluded,
                        limits=limits,
                        random_state=random_state)
        all_images.append(data_img)
        all_file_names.append(fnames)

    data = np.concatenate(all_images, axis=3)
    return data, label_data, all_file_names
