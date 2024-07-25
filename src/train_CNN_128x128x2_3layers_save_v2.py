"""
Train the model of CNN with 3 convolutional layers
on data  of size 128x128x2
read from a folder on basis of the train,in files.

Predict on val data, print the classification record.

If needed, save the model.
"""

# imports


import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report

from keras.utils import to_categorical
import time

# for loading data
from features.load_images_from_CSV_v1 import load_images

# specification of CNN:
from features.funcs_CNN_3layers import pred_mat2class, define_fit_CNN

# Hyperparamètres (réduction de l'image)

IMG_HEIGHT = 128
IMG_WIDTH = 128

batch_size = 128
epochs = 20

# Charger les images
path_data = "../data"
path_list_train_test = './models/datasets1_2'
df_train = pd.read_csv(os.path.join(path_list_train_test, 'img_train.csv'))
df_val = pd.read_csv(os.path.join(path_list_train_test, 'img_val.csv'))

X_train = load_images(path_data,
                      df_train,
                      new_size=(IMG_HEIGHT, IMG_WIDTH))
X_val = load_images(path_data,
                    df_val,
                    new_size=(IMG_HEIGHT, IMG_WIDTH))
# print(X_train.shape)  # (5400, 128, 128, 2)

y_train = df_train['num_class'].to_numpy()
y_val = df_val['num_class'].to_numpy()
print("------------")

# labels -> categorical
Y_train = to_categorical(y_train)
Y_val = to_categorical(y_val)

# Création, entraînement du modèle.
start = time.time()
model, __ = define_fit_CNN(X_train,
                           Y_train,
                           X_val,
                           Y_val,
                           summary=True)
done = time.time()

elapsed = done - start
print(f'Training : {elapsed:.2f} sec \n\n')
# 4 min


# Save model if needed
# dir_model = "../models"
# model.save(os.path.join(dir_model, "model2_1_3layers.keras"))
# 5 MB

# Performances du modèle

# Prédictions sur l'ensemble de validation.
start = time.time()
val_pred = model.predict(X_val)
done = time.time()

elapsed = done - start
print(f'Prediction : {elapsed:.2f} sec \n\n')
# 1.2 sec

print(val_pred[:3, :])  # The first predictions: 2, 2, 1

y_val_class = pred_mat2class(Y_val)
val_pred_class = pred_mat2class(val_pred)

print("Accuracy on validation:", accuracy_score(y_val_class, val_pred_class))
# 0.84, 0.83, 0.82, 0.84, 0.847, 0.824, 0.83, 0.807, 0.83

print(classification_report(y_val_class, val_pred_class))
# precision    recall  f1-score   support (Classe COVID)
# 0.84      0.83      0.83       450
# 0.81      0.85      0.83       450
# 0.80      0.78      0.79
# 0.84      0.83      0.83
# 0.84      0.85      0.85
# 0.86      0.74      0.80
# 0.78      0.86      0.82
# 0.80      0.76      0.78
# 0.83      0.81      0.82

# confusion matrix
print(pd.crosstab(y_val_class, val_pred_class))

# Prédictions sur l'ensemble train
start = time.time()
train_pred = model.predict(X_train)
done = time.time()

elapsed = done - start
print(f'Prediction : {elapsed:.2f} sec \n\n')

y_train_class = pred_mat2class(Y_train)
train_pred_class = pred_mat2class(train_pred)

print("Accuracy on train:", accuracy_score(y_train_class, train_pred_class))
# 0.87, 0.85, 0.83, 0.86, 0.86, 0.84, 0.85, 0.8378, 0.84
