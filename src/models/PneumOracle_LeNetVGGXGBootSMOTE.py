
import os
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.decomposition import PCA
import joblib
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, save_model, load_model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import xgboost as xgb
import matplotlib.pyplot as plt

# Paramètres
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
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

# Charger et traiter les images
print("Chargement et traitement des images...")
train_images = [load_image_and_mask(img, msk) for img, msk in zip(train_image_files, train_mask_files)]
train_images = [img for img in train_images si img is not None]  # Filtrer les images non chargées
X_train = np.array(train_images).reshape(len(train_images), -1)
print("Images chargées et traitées.")

# Appliquer SMOTE pour suréchantillonner les classes minoritaires
print("Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote = X_train_smote.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)
print("SMOTE appliqué.")
print("Répartition des classes dans l'ensemble d'entraînement après SMOTE:", dict(Counter(y_train_smote)))

# Convertir les étiquettes en encodage one-hot
print("Conversion des étiquettes en encodage one-hot...")
y_train_smote = to_categorical(y_train_smote, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)
print("Conversion terminée.")

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

# Modèle LeNet pour l'extraction de caractéristiques
print("Création du modèle LeNet pour l'extraction de caractéristiques...")
def lenet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation='relu'))
    model.add(GlobalAveragePooling2D())
    return model

lenet_model = lenet_model()
save_model(lenet_model, 'lenet_model.keras')  # Enregistrer le modèle LeNet pour une utilisation future
print("Modèle LeNet créé et enregistré.")

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

# Extraire les caractéristiques à l'aide de LeNet
print("Extraction des caractéristiques à l'aide de LeNet...")
lenet_train_features = extract_features(train_generator, lenet_model, train_steps, len(train_image_files))
lenet_val_features = extract_features(val_generator, lenet_model, val_steps, len(val_image_files))
print("Extraction des caractéristiques LeNet terminée.")

# Sortie de débogage
print(f'vgg_train_features shape: {vgg_train_features.shape}')
print(f'lenet_train_features shape: {lenet_train_features.shape}')
print(f'vgg_val_features shape: {vgg_val_features.shape}')
print(f'lenet_val_features shape: {lenet_val_features.shape}')

# Assurer la correspondance des dimensions
print("Vérification et ajustement des dimensions...")
min_train_samples = min(vgg_train_features.shape[0], lenet_train_features.shape[0])
min_val_samples = min(vgg_val_features.shape[0], lenet_val_features.shape[0])

vgg_train_features = vgg_train_features[:min_train_samples]
lenet_train_features = lenet_train_features[:min_train_samples]
y_train_smote = y_train_smote[:min_train_samples]

vgg_val_features = vgg_val_features[:min_val_samples]
lenet_val_features = lenet_val_features[:min_val_samples]
y_val = y_val[:min_val_samples]
print("Dimensions ajustées.")

# Combiner les caractéristiques
print("Combinaison des caractéristiques de VGG16 et LeNet...")
X_train = np.concatenate([vgg_train_features, lenet_train_features], axis=1)
X_val = np.concatenate([vgg_val_features, lenet_val_features], axis=1)
print("Combinaison des caractéristiques terminée.")

# Standardiser les caractéristiques
print("Standardisation des caractéristiques...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
print("Standardisation terminée.")

# Appliquer PCA
print("Application de la PCA...")
pca = PCA(n_components=0.95)  # Retenir 95% de la variance
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
print("PCA appliquée.")

# Enregistrer le modèle PCA
joblib.dump(pca, 'pca_model.pkl')
print("Modèle PCA enregistré.")
joblib.dump(scaler, 'scaler.pkl')

# Entraîner le modèle XGBoost
print("Entraînement du modèle XGBoost...")
dtrain = xgb.DMatrix(X_train_pca, label=np.argmax(y_train_smote, axis=1))
dval = xgb.DMatrix(X_val_pca, label=np.argmax(y_val, axis=1))

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

accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred)
class_report = classification_report(np.argmax(y_val, axis=1), y_pred, target_names=label_encoder.classes_)
auc_roc = roc_auc_score(np.argmax(y_val, axis=1), y_pred_proba, multi_class='ovr')
f1 = f1_score(np.argmax(y_val, axis=1), y_pred, average='weighted')

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
    lenet_model = load_model('lenet_model.keras')
    pca = joblib.load('pca_model.pkl')
    scaler = joblib.load('scaler.pkl')
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
            lenet_features = lenet_model.predict(np.expand_dims(img_normalized, axis=0))
            combined_features = np.concatenate([vgg_features, lenet_features], axis=1)

            combined_features_pca = pca.transform(combined_features)
            dimg = xgb.DMatrix(combined_features_pca)
            prediction_proba = xgb_model.predict(dimg)
            prediction = np.argmax(prediction_proba, axis=1)
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Classe Prédite : {predicted_class}')
            plt.axis('off')
            plt.show()

# Prédire et afficher les images de test
print("Prédiction et affichage des images de test...")
predict_and_display_images(test_image_dir)
print("Prédiction et affichage terminés.")






'''LeNet+VGG +SMOTE
Evaluating XGBoost model on validation data...
Accuracy: 0.7617233991906689
Classification Report:
                 precision    recall  f1-score   support

          COVID       0.79      0.30      0.43       697
   Lung Opacity       0.75      0.76      0.75      1177
         Normal       0.76      0.92      0.83      2065
Viral Pneumonia       0.81      0.79      0.80       262

       accuracy                           0.76      4201
      macro avg       0.78      0.69      0.70      4201
   weighted avg       0.76      0.76      0.74      4201

AUC-ROC: 0.9236279782746822
F1 Score: 0.7412737385173055'''