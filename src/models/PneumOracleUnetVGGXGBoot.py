import os
import numpy as np
import cv2
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb
import matplotlib.pyplot as plt

# Paramètres
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
TARGET_COUNT = 10000
LIMIT = 25000

base_dir = '/content/workspace/workspace'
image_dirs = [
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/COVID/images'),
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/NORMAL/images'),
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/LUNG_OPACITY/images'),
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/Viral Pneumonia/images')
]

mask_dirs = [
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/COVID/masks'),
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/NORMAL/masks'),
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/LUNG_OPACITY/masks'),
    os.path.join(base_dir, 'archive/COVID-19_Radiography_Dataset/Viral Pneumonia/masks')
]

test_image_dir = os.path.join(base_dir, 'Project/test_image')

labels = ['COVID', 'Normal', 'Lung Opacity', 'Viral Pneumonia']

# Fonction pour charger les images et les masques
def load_image_and_mask(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is not None and mask is not None:
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        img = img.astype(np.float16) / 255.0
        mask = mask.astype(np.float16) / 255.0
        img = img * np.expand_dims(mask, axis=-1)
        return img
    return None

# Fonction pour créer un générateur de données
def create_data_generator(image_files, mask_files, labels, batch_size, augment=False):
    data_gen_args = dict(rotation_range=15,
                         width_shift_range=0.15,
                         height_shift_range=0.15,
                         shear_range=0.15,
                         zoom_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest') if augment else {}

    data_gen = ImageDataGenerator(**data_gen_args)
    
    while True:
        for start in range(0, len(image_files), batch_size):
            end = min(start + batch_size, len(image_files))
            batch_images = []
            batch_labels = labels[start:end]
            for img_path, mask_path in zip(image_files[start:end], mask_files[start:end]):
                img = load_image_and_mask(img_path, mask_path)
                if img is not None:
                    batch_images.append(img)
            batch_images = np.array(batch_images)
            if len(batch_images) == 0:
                continue
            batch_images = data_gen.flow(batch_images, batch_size=len(batch_images), shuffle=False).next()
            yield batch_images, np.array(batch_labels)

def load_data(image_dirs, mask_dirs, labels, limit=None):
    image_files = []
    mask_files = []
    label_data = []
    for i, (image_dir, mask_dir) in enumerate(zip(image_dirs, mask_dirs)):
        print(f"Chargement des fichiers depuis {image_dir} et {mask_dir}")
        files = os.listdir(image_dir)
        if limit:
            files = files[:limit]
        for file_name in files:
            if file_name.endswith('.png'):
                img_path = os.path.join(image_dir, file_name)
                mask_path = os.path.join(mask_dir, file_name)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    image_files.append(img_path)
                    mask_files.append(mask_path)
                    label_data.append(labels[i])
    return image_files, mask_files, np.array(label_data)

# Charger les données
print("Chargement des données...")
image_files, mask_files, labels = load_data(image_dirs, mask_dirs, labels, limit=LIMIT)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print("Données chargées et étiquettes encodées.")

# Diviser les données en ensembles d'entraînement et de validation
print("Division des données en ensembles d'entraînement et de validation...")
train_image_files, val_image_files, train_mask_files, val_mask_files, y_train, y_val = train_test_split(
    image_files, mask_files, y, test_size=0.2, random_state=42
)
print("Données divisées.")

# Créer des générateurs de données
print("Création des générateurs de données...")
train_generator = create_data_generator(train_image_files, train_mask_files, y_train, batch_size=BATCH_SIZE, augment=True)
val_generator = create_data_generator(val_image_files, val_mask_files, y_val, batch_size=BATCH_SIZE, augment=False)
print("Générateurs de données créés.")

# Modèle VGG16 pour l'extraction de caractéristiques
print("Création du modèle VGG16 pour l'extraction de caractéristiques...")
vgg_base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
vgg_model = Model(inputs=vgg_base_model.input, outputs=GlobalAveragePooling2D()(vgg_base_model.output))
save_model(vgg_model, 'vgg_model.keras')  # Enregistrer le modèle VGG pour une utilisation future
print("Modèle VGG16 créé et enregistré.")

# Modèle UNET pour l'extraction de caractéristiques
print("Création du modèle UNET pour l'extraction de caractéristiques...")
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    return model

unet_base_model = unet_model()
unet_model = Model(inputs=unet_base_model.input, outputs=GlobalAveragePooling2D()(unet_base_model.output))
save_model(unet_model, 'unet_model.keras')  # Enregistrer le modèle UNET pour une utilisation future
print("Modèle UNET créé et enregistré.")

# Fonction pour extraire les caractéristiques en lots
def extract_features(generator, model, steps, expected_len):
    print(f"Extraction des caractéristiques en cours pour {expected_len} échantillons...")
    features = []
    count = 0
    for _ in range(steps):
        images, _ = next(generator)
        batch_features = model.predict(images)
        features.append(batch_features)
        count += len(batch_features)
        if count >= expected_len:
            break
    concatenated_features = np.concatenate(features)
    print(f"Extraction des caractéristiques terminée pour {count} échantillons.")
    return concatenated_features[:expected_len]

# Calculer les étapes par époque
train_steps = len(train_image_files) // BATCH_SIZE
val_steps = len(val_image_files) // BATCH_SIZE

# Extraire les caractéristiques à l'aide de VGG16
print("Extraction des caractéristiques à l'aide de VGG16...")
vgg_train_features = extract_features(train_generator, vgg_model, train_steps, len(train_image_files))
vgg_val_features = extract_features(val_generator, vgg_model, val_steps, len(val_image_files))
print("Extraction des caractéristiques VGG16 terminée.")

# Extraire les caractéristiques à l'aide de UNET
print("Extraction des caractéristiques à l'aide de UNET...")
unet_train_features = extract_features(train_generator, unet_model, train_steps, len(train_image_files))
unet_val_features = extract_features(val_generator, unet_model, val_steps, len(val_image_files))
print("Extraction des caractéristiques UNET terminée.")

# Sortie de débogage
print(f'vgg_train_features shape: {vgg_train_features.shape}')
print(f'unet_train_features shape: {unet_train_features.shape}')
print(f'vgg_val_features shape: {vgg_val_features.shape}')
print(f'unet_val_features shape: {unet_val_features.shape}')

# Assurer la correspondance des dimensions
print("Vérification et ajustement des dimensions...")
min_train_samples = min(vgg_train_features.shape[0], unet_train_features.shape[0])
min_val_samples = min(vgg_val_features.shape[0], unet_val_features.shape[0])

vgg_train_features = vgg_train_features[:min_train_samples]
unet_train_features = unet_train_features[:min_train_samples]
y_train = y_train[:min_train_samples]

vgg_val_features = vgg_val_features[:min_val_samples]
unet_val_features = unet_val_features[:min_val_samples]
y_val = y_val[:min_val_samples]
print("Dimensions ajustées.")

# Combiner les caractéristiques
print("Combinaison des caractéristiques de VGG16 et UNET...")
X_train = np.concatenate([vgg_train_features, unet_train_features], axis=1)
X_val = np.concatenate([vgg_val_features, unet_val_features], axis=1)
print("Combinaison des caractéristiques terminée.")

# Entraîner le modèle XGBoost
print("Entraînement du modèle XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'multi:softprob',
    'num_class': NUM_CLASSES,
    'eval_metric': 'mlogloss',
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42
}

evals = [(dtrain, 'train'), (dval, 'eval')]
xgb_model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50, verbose_eval=10)
print("Modèle XGBoost entraîné.")

# Évaluer le modèle XGBoost
print("Évaluation du modèle XGBoost sur les données de validation...")
y_pred_proba = xgb_model.predict(dval)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_val, y_pred)
class_report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
auc_roc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
f1 = f1_score(y_val, y_pred, average='weighted')

print(f'Précision : {accuracy}')
print(f'Rapport de classification :\n{class_report}')
print(f'AUC-ROC : {auc_roc}')
print(f'Score F1 : {f1}')

# Enregistrer les modèles et objets de prétraitement
print("Enregistrement du modèle XGBoost et des objets de prétraitement...")
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Modèles et objets enregistrés.")

# Fonction pour prétraiter et prédire une nouvelle image
def predict_and_display_images(test_image_dir):
    print("Chargement des modèles et objets de prétraitement pour la prédiction...")
    xgb_model = joblib.load('xgb_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    vgg_model = load_model('vgg_model.keras')
    unet_model = load_model('unet_model.keras')
    print("Modèles chargés pour la prédiction.")

    for file_name in os.listdir(test_image_dir):
        if file_name.endswith('.png'):
            img_path = os.path.join(test_image_dir, file_name)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"Erreur lors du chargement de l'image {file_name}")
                continue

            img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            img_normalized = img_resized.astype(np.float16) / 255.0

            vgg_features = vgg_model.predict(np.expand_dims(img_normalized, axis=0))
            unet_features = unet_model.predict(np.expand_dims(img_normalized, axis=0))
            combined_features = np.concatenate([vgg_features, unet_features], axis=1)

            dimg = xgb.DMatrix(combined_features)
            prediction_proba = xgb_model.predict(dimg)
            prediction = np.argmax(prediction_proba, axis=1)
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Fichier : {file_name} - Classe prédite : {predicted_class}')
            plt.axis('off')
            plt.show()

# Prédire et afficher les images de test
print("Prédiction et affichage des images de test...")
predict_and_display_images(test_image_dir)
print("Prédiction et affichage terminées.")


'''
Accuracy: 0.7771959057367294
Classification Report:
                 precision    recall  f1-score   support

          COVID       0.83      0.34      0.48       697
   Lung Opacity       0.75      0.78      0.76      1177
         Normal       0.78      0.92      0.84      2065
Viral Pneumonia       0.85      0.82      0.84       262

       accuracy                           0.78      4201
      macro avg       0.80      0.72      0.73      4201
   weighted avg       0.78      0.78      0.76      4201

AUC-ROC: 0.9346545539509379
F1 Score: 0.760038722864591'''