"""
Test example for the function predict_images_masks.

The data is loaded from the original folder in this script.
"""


# import tensorflow as tf
import os
import pandas as pd

# from features.load_masked_images_from_CSV import load_images
import features.load_masked_images_from_CSV as VGGload
import features.load_images_from_CSV_v2 as CNNload

from features.predict_from_tensor_v3 import predict_from_data, arr_keys_to_vals
# for confusion matrix .

from sklearn.metrics import accuracy_score

params = {
    'CNN': {'new_size': (128, 128),
            'file_model': "CNN_img_mask.keras",
            'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
            'load_img': CNNload.load_images},
    'VGG': {'new_size': (128, 128),
            'file_model': "VGG_img_mask_128.keras",
            'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
            'load_img': VGGload.load_images}
}

# parameters of VGG in a dict of 1 element.

if __name__ == "__main__":
    path_data = "../data"
    path_all_models = "../models/"
    path_datasets = "./datasets"

    print('----------')
    classifier = input("Quel modèle utiliser (choisissez: VGG ou CNN)?")

    path_model = os.path.join(path_all_models, params[classifier]['file_model'])

    path_list_files = os.path.join(path_datasets, 'val.csv')

    df_data = pd.read_csv(path_list_files)
    print("------------")
    # print(df_data['Images'])
    """
    Can be added for debugging!
    """

    print("------------")
    """
    X_val = VGGload.load_images(path_data,
                        df_data,
                        new_size=params[classifier]['new_size'])
    """
    X_val = params[classifier]['load_img'](path_data,
                                           df_data,
                                           new_size=params[classifier]['new_size'])

    print(X_val.shape)

    # Predictions as list of numbers.
    val_pred_class = predict_from_data(path_model,
                                       X_val,
                                       pred_labels=False)

    y_val = df_data['num_class']

    class_map = params[classifier]['class_map']
    y_val_names = arr_keys_to_vals(class_map, y_val)

    names_pred_class = arr_keys_to_vals(class_map, val_pred_class)

    # Print accuracy;
    val_acc_names = accuracy_score(y_val_names, names_pred_class)

    print("Accuracy:", val_acc_names)

    df_cm = pd.crosstab(y_val_names, names_pred_class)
    df_cm.index.set_names('ground_truth', inplace=True)
    df_cm.columns.set_names('predicted', inplace=True)
    print(df_cm)
