
# imports

import pandas as pd
import os
import time
from sklearn.metrics import accuracy_score, classification_report, f1_score
from keras.utils import to_categorical
import sys
sys.path.append('../features/')
# for loading data
from features.load_masked_images_from_CSV import load_images

# specification of VGG16:
from features.funcs_VGG16 import pred_mat2class, define_fit_VGG16

# Hyperparamètres (réduction de l'image)
IMG_WIDTH = IMG_HEIGHT = 256
batch_size = 64
epochs = 50

# Chargement des images
path_data = "../data"
path_list_train_test = './datasets'
df_train = pd.read_csv(os.path.join(path_list_train_test, 'train.csv'))
df_val = pd.read_csv(os.path.join(path_list_train_test, 'val.csv'))
df_test = pd.read_csv(os.path.join(path_list_train_test, 'test.csv'))

X_train = load_images(path_data,
                      df_train,
                      new_size=(IMG_HEIGHT, IMG_WIDTH))
X_val = load_images(path_data,
                    df_val,
                    new_size=(IMG_HEIGHT, IMG_WIDTH))
X_test = load_images(path_data,
                    df_test,
                    new_size=(IMG_HEIGHT, IMG_WIDTH))

# Définition et dichotomisation des variables de labels :
y_train = df_train['num_class'].to_numpy()
y_val = df_val['num_class'].to_numpy()
y_test = df_test['num_class'].to_numpy()

# labels -> categorical
Y_train = to_categorical(y_train)
Y_val = to_categorical(y_val)
Y_test = to_categorical(y_test)
# 0 : Normal
# 1 : Pulmonary infection
# 2 : Covid

# Création, entraînement du modèle.
start = time.time()
model, history = define_fit_VGG16(X_train,
                           Y_train,
                           X_val,
                           Y_val,
                            n_retrained_vgg_layers = 4,
                            batch_size=batch_size,
                            epochs=epochs,
                           summary=True)
done = time.time()
elapsed_min = np.floor((done - start)/60)
elapsed_s = done - start - (elapsed_min*60)

print(f'Training : {elapsed_min:.0f} min et {elapsed_s:.0f} s \n\n')

# Performances en prédiction sur l'ensemble de validation.
start = time.time()
val_pred = model.predict(X_val)
done = time.time()

elapsed_min = np.floor((done - start)/60)
elapsed_s = done - start - (elapsed_min*60)

print(f'Prediction on validation sample: {elapsed_min:.0f} min et {elapsed_s:.0f} s \n\n')

true_class_val = pred_mat2class(Y_val)
pred_class_val = pred_mat2class(val_pred)

print("Accuracy on validation:", accuracy_score(true_class_val, pred_class_val))
print("f1-score on validation:", f1_score(true_class_val, pred_class_val, average='macro'))

print(classification_report(true_class_val, pred_class_val)) 

# 43/43 [==============================] - 270s 6s/step
# Prediction on validation sample: 4 min et 33 s 


# Accuracy on validation: 0.845925925925926
# f1-score on validation: 0.8451857500091052
#               precision    recall  f1-score   support

#   (Normal) 0       0.79      0.93      0.85       450
#     (PI)   1       0.93      0.75      0.83       450
#    (Covid) 2       0.84      0.86      0.85       450

#     accuracy                           0.85      1350
#    macro avg       0.85      0.85      0.85      1350
# weighted avg       0.85      0.85      0.85      1350


# Performances en prédiction sur l'ensemble de test.

y_test_pred = model.predict(X_test)
true_class_test = pred_mat2class(Y_test)
pred_class_test = pred_mat2class(y_test_pred)
print("Accuracy on test:", accuracy_score(true_class_test, pred_class_test))
print("f1-score on test:", f1_score(true_class_test, pred_class_test, average='macro'))

print(classification_report(true_class_test, pred_class_test))
# 24/24 [==============================] - 154s 6s/step
# Accuracy on test: 0.8453333333333334
# f1-score on test: 0.8449204638125843
#               precision    recall  f1-score   support

#            0       0.78      0.94      0.85       250
#            1       0.93      0.76      0.84       250
#            2       0.85      0.84      0.84       250

#     accuracy                           0.85       750
#    macro avg       0.85      0.85      0.84       750
# weighted avg       0.85      0.85      0.84       750

# Enregistrement du modèle entraîné
model.save(f"../../models/VGG_img_mask_{IMG_WIDTH}.keras")


