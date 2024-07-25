"""
Test example for the function predict_images_masks.

A CSV file with 10 pairs image-mask is used.
The data is loaded from the original folder in this script.

"""


# import tensorflow as tf
# import os
import pandas as pd
# import cv2
# import numpy as np

from features.load_images_from_CSV_v1 import load_images
from features.predict_from_tensor import predict_from_data


IMG_HEIGHT = 128
IMG_WIDTH = 128
# class_map = {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'}


if __name__ == "__main__":
    path_data = "../data"
    path_model_3layers = "../models/model1_3layers.keras"
    path_list_files = './models/data_test1/data_test1.csv'
    df_data = pd.read_csv(path_list_files)
    print("------------")
    print(df_data['Images'])
    """
    ['COVID-10.png', 'COVID-100.png', 'COVID-20.png', 'COVID-42.png',
    'Lung_Opacity-42.png',
    'Normal-10.png', 'Normal-100.png', 'Normal-20.png', 'Normal-42.png',
    'Viral Pneumonia-42.png']

    """

    print("------------")

    tens_data = load_images(path_data,
                            df_data,
                            new_size=(IMG_HEIGHT, IMG_WIDTH))
    print(tens_data.shape)

    lst_predictions = predict_from_data(path_model_3layers,
                                        tens_data)

    print(lst_predictions)
    # Predictions , 3 mistakes out of 10:
    # ['Non-COVID', 'COVID', 'COVID', 'COVID',
    # 'COVID',
    # 'Normal', 'Normal', 'Normal', 'Normal',
    # 'Normal']
