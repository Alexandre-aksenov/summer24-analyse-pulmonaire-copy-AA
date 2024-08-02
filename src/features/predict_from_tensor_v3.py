"""
Function for testing a model.

path to model,
data,
->
predictions.

This function can be called after the images are loaded
in appropriate dimension.
"""


import tensorflow as tf
import numpy as np


def arr_keys_to_vals(dct, arr_keys):
    """
    Converts an iterable of keys to the corresponding values
    in the dictionary dct.
    """
    return [dct[__] for __ in arr_keys]


class_map = {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'}


def predict_from_data(path_model: str,
                      data: np.ndarray,
                      pred_labels=True):
    """
    Function for testing a model, which takes image and mask as input.

    Input.
        path_model: path to model,
        data : np.ndarray
            (num_img x WIDTH x HEIGHT x 1 or 2 depending on tratement of masks)
        pred_labels; output string labels (True, default)
            or indices of classes (False).

    Returns:
        lst_predictions : list of strings or 1darray.
    """

    model = tf.keras.models.load_model(path_model)
    prediction_proba = model.predict(data)
    predicted_class = np.argmax(prediction_proba, axis=-1)

    if (pred_labels):
        return arr_keys_to_vals(class_map, predicted_class)
    else:
        return predicted_class


# test : see the script 'pred_dir_img_mask_10img_from_CSV_v4'
