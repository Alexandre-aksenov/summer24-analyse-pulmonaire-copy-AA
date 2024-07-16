"""
Functions for loading a part of images
in lower resolution
and as black-and-white.
"""

import os
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import cv2


def list_img(image_dir: str, sought_ext='.png'):
    """
    List of files with a specific extension in a folder.

    Useful for extracting the number of files.
    Can also be used as the first step of the next version of function 'load_images'.

    Args:
        image_dir (str): path to folder
        sought_ext (str, optional): sought extension. Defaults to '.png'.

    Returns:
        list of str: the list of image files in the order of os.listdir.
    """
    files = os.listdir(image_dir)
    lst_images = [file_name for file_name in files if file_name.endswith(sought_ext)]
    return lst_images


def load_images(image_dir, label, new_size=(28, 28), limit=None, random_state=None):
    """
    Fonction pour charger les images d'un dossier.
    
    Args:
        image_dir (str): path to folder.
        label (str or int): the label to associate to the images.
        new_size (tuple (int, int)): the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).
        limit (int, optional): the max number of files to load.
            Defaults to None (load all PNG images).
        random_state (None or int): controls the random generator.

    Called by the function: load_data.

    Returns:
        tensor NImages x HEIGHT x WIDTH x 1: the images resized to HEIGHT x WIDTH;
        1d array (NImages,) : the label expanded to array.
    
    Possible improvements.
    1. Speed up the function by preallocating the tensor first,
    then writing into it.

    4. Return the  filenames of the extracted images as a 3rd output.
    """
    image_data = []
    label_data = []
    rng = np.random.default_rng(seed=random_state)

    all_files = list_img(image_dir)
    
    if limit:
        files = rng.choice(all_files, size=limit, replace=False)
    else: # adeed in comparison to V2.
        files = all_files
        
    for file_name in files:
        img_path = os.path.join(image_dir, file_name)
        try:
            # Read the grayscale image as: HEIGHT x WIDTH x 1
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = cv2.resize(img, new_size)
            img = np.expand_dims(img, axis=-1)
            
            img = img / 255.0
            image_data.append(img)
            label_data.append(label)
        except Exception as e:
            print(f"Erreur de chargement de l'image {file_name} : {e}")
    
    return np.array(image_data), np.array(label_data)


def load_data(image_dirs, labels, new_size=(28, 28), limits=None, random_state=None):
    """
    Load data from several folders

    Args:
        image_dirs (list of 'str'): the paths to folders.
        labels (list of 'str' of the same length): labels to associate to each directory.
        new_size (tuple (int, int)): the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).
        limits (list if int of None, optional): the number of files to load.
            Defaults to None.
        random_state (None or int): controls the random generator.

    Returns:
        tensor sum_{folders}(NImages) x HEIGHT x WIDTH :
            the images resized to HEIGHT x WIDTH;
        1d array (sum_{folders}(NImages),) : the labels expanded to array.
    """
    all_images = []
    all_labels = []
    
    limits2 = [None] * len(image_dirs) if limits is None else limits
    
    for i, image_dir in enumerate(image_dirs):
        images, label_data = load_images(image_dir,
                                        labels[i],
                                        limit=limits2[i],
                                        new_size=new_size,
                                        random_state=random_state)
        all_images.append(images)
        all_labels.append(label_data)
    return np.concatenate(all_images), np.concatenate(all_labels)


def load_data_masks(image_dirs, labels, new_size=(28, 28), limits=None, random_state=None):
    """
    Load data from several folders grouped as a list of lists.

    Args:
        image_dirs (list of lists of 'str'): the paths to folders.
            The sublists should be of same length and represent different modalities
            of observation of same individuals.
        
        labels (list of 'str' of the same length as sublists above): labels to associate to each directory.
        new_size (tuple (int, int)): the size the images are reshaped to when loading,
            interpreted as (IMG_WIDTH, IMG_HEIGHT).
        limits (list if int of None, optional): the number of files to load.
            Defaults to None.
        random_state (None or int): controls the random generator.

    Returns:
        tensor sum_{folders}(NImages) x HEIGHT x WIDTH x num_modalities:
            the images resized to HEIGHT x WIDTH;
        1d array (sum_{folders}(NImages),) : the labels expanded to array.
    """
    all_images = []
    for idx, modality in enumerate(image_dirs):
        if (idx == 0):
            data_img, label_data = load_data(modality,
                        labels,
                        new_size=new_size,
                        limits=limits,
                        random_state=random_state)
        else:
            data_img, __ = load_data(modality,
                        labels,
                        new_size=new_size,
                        limits=limits,
                        random_state=random_state)
        all_images.append(data_img)
        
    data = np.concatenate(all_images, axis=3)
    return data, label_data
