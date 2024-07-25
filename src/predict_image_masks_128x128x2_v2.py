"""
Function for testing a model.

path to model,
path to data,
path to masks
->
predictions.

This function assumes thet the images are loaded in resolution 128x128x2
in the imported module.

"""


# import tensorflow as tf
# import os
# import cv2
import numpy as np

from features.load_images_limit_masks_v3 import list_img, load_data_masks
from features.predict_from_tensor import predict_from_data


IMG_HEIGHT = 128
IMG_WIDTH = 128
class_map = {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'}


def predict_images_masks(path_model: str,
                         test_image_dir: str,
                         test_mask_dir: str):
    """
    Function for testing a model, which takes image and mask as input.

    Input.
        path to model,
        test_image_dir: path to images,
        test_mask_dir: path to masks.

    Returns:
        lst_predictions : list of strings
    """

    # model = tf.keras.models.load_model(path_model)

    num_img = len(list_img(test_image_dir))
    dummy_labels = -1 * np.ones(num_img, dtype=int)

    data, __, __ = load_data_masks([[test_image_dir], [test_mask_dir]],
                                   dummy_labels,
                                   new_size=(IMG_WIDTH, IMG_HEIGHT))

    # printout for debugging
    # print(data.shape)  # (10, 128, 128, 2)

    predicted_class_label = predict_from_data(path_model, data)
    return predicted_class_label

# test : see the script 'pred_dir_img_mask_v04'.
