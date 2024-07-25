"""
Test example for the function predict_images_masks.

A folder with 10 pairs image-mask is used.
"""


# import tensorflow as tf
import os
# import cv2
# import numpy as np

from predict_image_masks_128x128x2_v2 import predict_images_masks


# IMG_HEIGHT = 128
# IMG_WIDTH = 128
# class_map = {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'}


if __name__ == "__main__":
    path_data = "./models/data_test1"
    path_img = os.path.join(path_data, "images")
    path_masks = os.path.join(path_data, "masks")
    path_model_3layers = "../models/model1_3layers.keras"
    sorted_lst_files = sorted(os.listdir(path_img))
    print("------------")
    print(sorted_lst_files)
    """
    ['COVID-10.png', 'COVID-100.png', 'COVID-20.png', 'COVID-42.png',
    'Lung_Opacity-42.png',
    'Normal-10.png', 'Normal-100.png', 'Normal-20.png', 'Normal-42.png',
    'Viral Pneumonia-42.png']

    """

    print("------------")

    lst_predictions = predict_images_masks(path_model_3layers,
                                           path_img,
                                           path_masks)

    print(lst_predictions)
    # Predictions , 3 mistakes out of 10:
    # ['Non-COVID', 'COVID', 'COVID', 'COVID',
    # 'COVID',
    # 'Normal', 'Normal', 'Normal', 'Normal',
    # 'Normal']
