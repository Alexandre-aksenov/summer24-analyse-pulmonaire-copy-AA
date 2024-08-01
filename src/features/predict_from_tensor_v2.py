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

# from features.load_images_limit_masks_v3 import list_img, load_data_masks


# IMG_HEIGHT = 128
# IMG_WIDTH = 128
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

    # printout for debugging
    # print(data.shape)  # (10, 128, 128, 2)

    prediction_proba = model.predict(data)
    # print(prediction_proba.shape)  # (10, 3)

    predicted_class = np.argmax(prediction_proba, axis=-1)
    # print(predicted_class.shape)  # (10,)

    # predicted_class_label = [class_map[__] for __ in predicted_class]
    # return predicted_class_label

    if (pred_labels):
        return [class_map[__] for __ in predicted_class]
    else:
        return predicted_class


# test : see the script 'pred_dir_img_mask_10img_from_CSV_v3' TOWRITE
