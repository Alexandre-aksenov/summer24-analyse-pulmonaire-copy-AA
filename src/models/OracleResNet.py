import os
import subprocess
import sys
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Installer les paquets requis si non déjà installés
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import absl
except ImportError:
    install('absl-py')
    import absl

# Paramètres
IMG_HEIGHT = 300  # Taille augmentée de l'image pour plus de détails
IMG_WIDTH = 300   # Taille augmentée de l'image pour plus de détails
NUM_CLASSES = 4
BATCH_SIZE = 8  # Ajuster selon la mémoire GPU
EPOCHS = 35  # Nombre d'époques d'entraînement
IMAGE_LIMIT = 2000  # Limite du nombre d'images traitées par classe

# Définir les répertoires d'images et de masques
base_dir = '/content/workspace/workspace'
base_dir_test = '/content/Project'
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
test_image_dir = '/content/Project/test_image/image'
test_mask_dir = '/content/Project/test_image/mask'

labels = ['COVID', 'Normal', 'Lung Opacity', 'Viral Pneumonia']

# Dictionnaire pour mapper les étiquettes de classe numériques à leurs noms
class_map = {0: 'COVID', 1: 'Normal', 2: 'Lung Opacity', 3: 'Viral Pneumonia'}

# Fonction pour charger les images et les masques
def load_images_and_masks(image_dir, mask_dir, label, limit=None):
    image_data = []
    label_data = []
    files = os.listdir(image_dir)
    if limit:
        files = files[:limit]
    for file_name in files:
        if file_name.endswith('.png'):
            img_path = os.path.join(image_dir, file_name)
            mask_path = os.path.join(mask_dir, file_name)
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"File not found: {img_path} or {mask_path}")
                continue
            try:
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    print(f"Error loading image or mask {file_name}")
                    continue
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
                img = img.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0
                img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))  # Superposer le masque sur l'image
                image_data.append(img)
                label_data.append(label)
            except Exception as e:
                print(f"Error loading image or mask {file_name}: {e}")
    return np.array(image_data), np.array(label_data)

# Fonction pour charger les données de plusieurs répertoires
def load_data(image_dirs, mask_dirs, labels, limit=None):
    all_images = []
    all_labels = []
    for i in range(len(image_dirs)):
        images, label_data = load_images_and_masks(image_dirs[i], mask_dirs[i], i, limit)
        all_images.append(images)
        all_labels.append(label_data)
    return np.concatenate(all_images), np.concatenate(all_labels)

# Charger et prétraiter les données
print("Chargement et prétraitement des données...")
images, labels = load_data(image_dirs, mask_dirs, labels, limit=IMAGE_LIMIT)

# Encoder les labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Diviser les données en ensembles d'entraînement et de test
print("Division des données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
print(f"Ensemble d'entraînement : {X_train.shape}, Ensemble de test : {X_test.shape}")

# Génération des données avec Data Augmentation
print("Génération des données avec Data Augmentation...")
data_gen_args = dict(rotation_range=15,
                     width_shift_range=0.15,
                     height_shift_range=0.15,
                     shear_range=0.15,
                     zoom_range=0.15,
                     horizontal_flip=True,
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)

# Utiliser la même graine pour les générateurs d'images et de masques
seed = 42
image_datagen.fit(X_train, augment=True, seed=seed)

train_image_generator = image_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=seed)

# Générateur pour les données de validation
val_image_datagen = ImageDataGenerator()
val_image_generator = val_image_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, seed=seed)

# Définir le modèle ResNet50 sans couche de convolution supplémentaire
def resnet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = Input(input_size)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Créer et compiler le modèle ResNet50
print("Création et compilation du modèle ResNet50...")
model = resnet_model()

# Planificateur de taux d'apprentissage
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Ajouter des callbacks pour l'arrêt précoce et la réduction du taux d'apprentissage
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6),
    LearningRateScheduler(scheduler)
]

# Entraîner le modèle ResNet50
print("Entraînement du modèle ResNet50...")
history = model.fit(train_image_generator, epochs=EPOCHS, validation_data=val_image_generator, callbacks=callbacks)

# Sauvegarder le modèle
print("Sauvegarde du modèle...")
model.save('resnet_model.keras')

# Évaluer le modèle
print("Évaluation du modèle...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Calculer le F1-score
print("Calcul du F1-score...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)
f1 = f1_score(y_test, y_pred_classes, average='weighted')
print(f'F1-score: {f1}')

# Afficher le rapport de classification
print("Rapport de classification :")
target_names = [class_map[i] for i in range(NUM_CLASSES)]
print(classification_report(y_test, y_pred_classes, target_names=target_names))

# Fonction pour prétraiter et prédire une nouvelle image
def predict_and_display_images(test_image_dir, test_mask_dir):
    model = tf.keras.models.load_model('resnet_model.keras')

    for file_name in os.listdir(test_image_dir):
        if file_name.endswith('.png'):
            img_path = os.path.join(test_image_dir, file_name)
            mask_path = os.path.join(test_mask_dir, file_name)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            mask_resized = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
            img_normalized = img_resized / 255.0
            mask_normalized = mask_resized / 255.0
            img_bitwise = cv2.bitwise_and(img_normalized, img_normalized, mask=mask_normalized.astype(np.uint8))
            img_expanded = np.expand_dims(img_bitwise, axis=0)

            prediction_proba = model.predict(img_expanded)
            predicted_class = np.argmax(prediction_proba, axis=-1)[0]
            predicted_class_label = class_map[predicted_class]

            # Afficher l'image avec le masque superposé
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor((img_bitwise * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.title(f'File: {file_name} - Predicted Class: {predicted_class} ({predicted_class_label})')
            plt.axis('off')

            # Afficher le masque original
            plt.subplot(1, 2, 2)
            plt.imshow(mask_resized, cmap='viridis')
            plt.title(f'Mask: {file_name}')
            plt.axis('off')

            plt.show()

import gc
# Prédire et afficher les images de test
print("Prédiction et affichage des images de test...")
predict_and_display_images(test_image_dir, test_mask_dir)
del images, labels, X_train, X_test, y_train, y_test
gc.collect()
