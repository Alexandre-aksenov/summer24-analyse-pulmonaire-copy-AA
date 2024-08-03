# Importation functions
import os
import features.load_images_from_CSV_v2 as CNNload
import features.load_masked_images_from_CSV as VGGload
# Model definition
from features.funcs_VGG16 import define_VGG16model
from features.funcs_CNN_3layers import define_CNN_3layers_depth2
import pandas as pd
import features.funcs_gradcam as gradcam
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt

params = {
    'CNN': {'new_size': (128, 128),
            'file_model': "CNN_weights.h5",
            'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
            'load_img': CNNload.load_img_mask,
            'model_specs' : define_CNN_3layers_depth2,
            'last_conv_layer_name' : "conv3",
            'make_gradcam_heatmap' :gradcam.make_gradcam_heatmap_CNN},
    'VGG': {'new_size': (128, 128),
            'file_model': "VGG16_weights.h5",
            'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
            'load_img': VGGload.load_img_mask,
            'model_specs' : define_VGG16model,
            'last_conv_layer_name' : "block5_conv3",
            'make_gradcam_heatmap' :gradcam.make_gradcam_heatmap_VGG16},
}

path_data = "../data"
path_datasets = './datasets/test_gradcam.csv'
path_all_models = "../models/"


print('---------- Sélection du modèle     ----------')

classifier = input("Quel modèle utiliser (choisissez: VGG ou CNN)?")

path_model = os.path.join(path_all_models, params[classifier]['file_model'])
img_size = params[classifier]['new_size']
# Make model
model = params[classifier]['model_specs'](*img_size,
                              num_classes = 3,
                              summary=True)
model.load_weights(path_model)
print(f"-----------Modèle sélectionné : {classifier} -----------")

df_data = pd.read_csv(path_datasets, index_col=0)

print("---------- Sélection de l'image ----------")

idx_im = int(input("Sur quelle image tester le modèle ? (Entrez un numéro entre 0 et 5)"))
row_data = df_data.iloc[idx_im, :]
scan_path = os.path.join(path_data, row_data['img_folder'], row_data['Images'])
mask_path = os.path.join(path_data, row_data['mask_folder'], row_data['Masks'])
img_path = (scan_path, mask_path)
print(f"-----------Image sélectionnée : {idx_im}-----------")

display(Image(img_path[0]))
print("Classe réelle :",row_data['num_class'])

# Prepare image
img_array = params[classifier]['load_img'](img_path, new_size=img_size)

# Remove last layer's softmax
model.layers[-1].activation = None

# Print what the top predicted class is
# preds = model.predict(img_array)
# print("Predicted:", preds)

# Generate class activation heatmap
last_conv_layer_name = params[classifier]['last_conv_layer_name']
heatmap_function = params[classifier]['make_gradcam_heatmap']
heatmap = heatmap_function(img_array, model, last_conv_layer_name)

# Display results
gradcam.save_and_display_gradcam(img_path, heatmap['heatmap'])
print("Classe prédite :", heatmap["class_id"])