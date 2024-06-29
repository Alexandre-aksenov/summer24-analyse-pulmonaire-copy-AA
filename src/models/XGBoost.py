import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import joblib

# Paramètres
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CLASSES = 4  # Nombre de classes mis à jour

# Définir les répertoires d'images
image_dirs = [
    r'C:/Users/khale/OneDrive/Documents/workspace/archive/COVID-19_Radiography_Dataset/COVID/images',
    r'C:/Users/khale/OneDrive/Documents/workspace/archive/COVID-19_Radiography_Dataset/Normal/images',
    r'C:/Users/khale/OneDrive/Documents/workspace/archive/COVID-19_Radiography_Dataset/Lung_Opacity/images',
    r'C:/Users/khale/OneDrive/Documents/workspace/archive/COVID-19_Radiography_Dataset/Viral Pneumonia/images'
]
labels = ['COVID', 'Normal', 'Lung Opacity', 'Viral Pneumonia']

# Fonction pour charger les images
def load_images(image_dir, label, limit=None):
    image_data = []
    label_data = []
    files = os.listdir(image_dir)
    if limit:
        files = files[:limit]
    for file_name in files:
        if file_name.endswith('.png'):
            img_path = os.path.join(image_dir, file_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                img = img / 255.0
                image_data.append(img)
                label_data.append(label)
            except Exception as e:
                print(f"Erreur de chargement de l'image {file_name} : {e}")
    return np.array(image_data), np.array(label_data)

# Fonction pour charger les données de plusieurs répertoires
def load_data(image_dirs, labels, limit=None):
    all_images = []
    all_labels = []
    for i, image_dir in enumerate(image_dirs):
        images, label_data = load_images(image_dir, labels[i], limit)
        all_images.append(images)
        all_labels.append(label_data)
    return np.concatenate(all_images), np.concatenate(all_labels)

# Charger et prétraiter les données
images, labels = load_data(image_dirs, labels, limit=25000)

# Aplatir les images
X = images.reshape(images.shape[0], -1)
y = labels

# Encoder les étiquettes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Calculer les poids des classes
class_counts = np.bincount(y)
total_samples = len(y)
class_weights = {i: total_samples / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)}
train_weights = np.array([class_weights[i] for i in y])

print(f"Poids des classes calculés : {class_weights}")

# Diviser les données
X_train, X_test, y_train, y_test, train_weights, test_weights = train_test_split(X, y, train_weights, test_size=0.2, random_state=42)

# Mettre à l'échelle les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Appliquer PCA
pca = PCA(n_components=75, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convertir les données en DMatrix
dtrain = xgb.DMatrix(X_train_pca, label=y_train, weight=train_weights)
dtest = xgb.DMatrix(X_test_pca, label=y_test, weight=test_weights)

# Définir les paramètres du modèle XGBoost
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

# Entraîner le modèle XGBoost avec early stopping
evals = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50)

# Enregistrer les modèles et objets de prétraitement
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(pca, 'pca.pkl')

# Évaluer le modèle XGBoost
y_pred_proba = xgb_model.predict(dtest)
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print(f'accuracy : {accuracy}')
print(f'Rapport de classification :\n{class_report}')
print(f'AUC-ROC : {auc_roc}')

# Fonction pour prétraiter et prédire une nouvelle image
def predict_and_display_images(test_image_dir):
    xgb_model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    pca = joblib.load('pca.pkl')

    for file_name in os.listdir(test_image_dir):
        if file_name.endswith('.png'):
            img_path = os.path.join(test_image_dir, file_name)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            img_normalized = img_resized / 255.0
            img_flattened = img_normalized.reshape(1, -1)

            img_scaled = scaler.transform(img_flattened)
            img_pca = pca.transform(img_scaled)

            dimg = xgb.DMatrix(img_pca)
            prediction_proba = xgb_model.predict(dimg)
            prediction = np.argmax(prediction_proba, axis=1)
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            plt.figure()
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Fichier : {file_name} - Classe prédite : {predicted_class}')
            plt.axis('off')
            plt.show()

# Répertoire contenant les images de test
test_image_dir = r'C:/Users/khale/OneDrive/Documents/workspace/Project/test_image'

# Prédire et afficher les images de test
predict_and_display_images(test_image_dir)
