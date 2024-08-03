"""
Prediction of validation data,
where the user is prompted to choose between 2 models.

The data is loaded from the original folder in this script
and the models are loaded as files of weights.

Function for prediction using a specified classifier
and data.

Used in the script 'src/pred_val_2models_weights.py'
and for use in presentation.
"""


import os
import pandas as pd

# import load_masked_images_from_CSV as VGGload
import features.load_masked_images_from_CSV as VGGload
import features.load_images_from_CSV_v2 as CNNload

from features.predict_from_tensor_weights import predict_from_data, arr_keys_to_vals

# for confusion matrix .
from sklearn.metrics import accuracy_score

# functions to define architecture
from features.funcs_CNN_3layers import define_CNN_3layers_depth2
from features.funcs_VGG16 import define_VGG16model


params = {
    'CNN': {'new_size': (128, 128),
            'file_model': "CNN_weights.h5",  # -> weights
            'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
            'load_img': CNNload.load_images,
            'def_archi': define_CNN_3layers_depth2},
    'VGG': {'new_size': (128, 128),
            'file_model': "VGG16_weights.h5",  # -> weights
            'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
            'load_img': VGGload.load_images,
            'def_archi': define_VGG16model}
}
# parameters of both models in a dict.

# values of paths (parameters) in the testing script
# 'pred_val_2models_weights'.
# path_data = "../data"
# path_all_models = "../models/"
# path_datasets = "./datasets"
# path_list_files = os.path.join(path_datasets, 'val.csv')

# -> parameter
# classifier = input("Quel modèle utiliser (choisissez: VGG ou CNN)?")


# if __name__ == "__main__":
def pred_weights_img(classifier,
                     path_list_files,
                     path_data="../data",
                     path_all_models="../models/"):
    """
    Function for prediction using a specified classifier
    and data.
    """

    df_data = pd.read_csv(path_list_files)
    # print("------------")
    # print(df_data['Images'])
    """
    Can be added for debugging!
    """

    X_val = params[classifier]['load_img'](path_data,
                                           df_data,
                                           new_size=params[classifier]['new_size'])

    print(X_val.shape)

    # Predictions as list of numbers. Adapted to loading weights.
    pth_model = os.path.join(path_all_models, params[classifier]['file_model'])
    val_pred_class = predict_from_data(pth_model,
                                       params[classifier]['def_archi'],
                                       X_val,
                                       pred_labels=False)

    y_val = df_data['num_class']

    class_map = params[classifier]['class_map']
    y_val_names = arr_keys_to_vals(class_map, y_val)

    names_pred_class = arr_keys_to_vals(class_map, val_pred_class)

    # Return accuracy;
    val_acc_names = accuracy_score(y_val_names, names_pred_class)

    # print("Accuracy:", val_acc_names)

    df_cm = pd.crosstab(y_val_names, names_pred_class)
    df_cm.index.set_names('ground_truth', inplace=True)
    df_cm.columns.set_names('predicted', inplace=True)
    # Return df_cm.
    return y_val_names, val_acc_names, df_cm
