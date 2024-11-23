"""
La 2e partie de la présentation du 5 août 2024.
La présentation était faite en deux parties.
Cette partie contient les sections
"Contexte", "Données", "Répartition par type de patient", "Répartition des sources"
(au début de la présentation)
et "Modélisation", "Démo", "Conclusion et perspectives"
à la fin de la présentation générale.

"""

import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import sys
sys.path.append('..')

# load an image
import features.load_images_from_CSV_v2 as CNNload
import features.load_masked_images_from_CSV as VGGload
# Model definition
from features.funcs_VGG16 import define_VGG16model
from features.funcs_CNN_3layers import define_CNN_3layers_depth2

import features.funcs_gradcam_streamlit as gradcam


from features.predict2models_weights import pred_weights_img


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Medical Image Data Exploration",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Fonction pour afficher les images
def display_images(dir_images, n_im=10):
    filenames = [name for name in os.listdir(dir_images)[:n_im]]
    images = [cv2.imread(os.path.join(dir_images, filename)) for filename in filenames]

    fig, axes = plt.subplots(1, n_im, figsize=(10, 2.5))  # Taille réduite de 125%
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)


# Chargement des données
data_folder = "../../data"

conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID", "Normal"]
lst_dirs_condition = [os.path.join(data_folder, cond, "images") for cond in conditions]
list_dir_images = pd.DataFrame({"img_type": ["Viral Pneumonia", "Lung Opacity", "Covid", "Normal"],
                                "img_dir": lst_dirs_condition})


# Fonction pour charger les métadonnées
@st.cache_data
def load_metadata():
    covid_meta = pd.read_excel(os.path.join(data_folder, "COVID.metadata.xlsx"))
    lung_opa_meta = pd.read_excel(os.path.join(data_folder, "Lung_Opacity.metadata.xlsx"))
    pneumonia_meta = pd.read_excel(os.path.join(data_folder, "Viral Pneumonia.metadata.xlsx"))
    normal_meta = pd.read_excel(os.path.join(data_folder, "Normal.metadata.xlsx"))
    data_meta = pd.concat([covid_meta, lung_opa_meta, pneumonia_meta, normal_meta])

    # Renommage des variables
    new_names = {
        'FILE NAME': 'file_name',
        'FORMAT': 'format',
        'SIZE': 'resolution',
        'URL': 'url'
    }
    data_meta.rename(columns=new_names, inplace=True)
    data_meta['data_type'] = data_meta.file_name.str.split('-').str[0]
    data_meta['data_type'].replace({
        "Lung_Opacity": "Lung Opacity",
        "COVID": "Covid",
        "NORMAL": "Normal"
    }, inplace=True)
    return data_meta


data_meta = load_metadata()

# Créer le menu de navigation
with st.sidebar:
    selected = option_menu("Menu",
        [
            "Contexte", "Données", "Répartition par type de patient",
            "Répartition des sources",
            "Recherche d’outliers", "Indicateurs de lisibilité des images",  # vide
            "Formalisation du modèle", "Modèles préselectionnés",  # vide
            "Modélisation", "Démo", "Conclusion et perspectives"],
        menu_icon="cast", default_index=0)

if selected == "Contexte":
    st.markdown("<h2 style='text-align: center;'>Radiopy-19</h2>", unsafe_allow_html=True)

    # Chemin vers l'image
    # image_path = "C:/Users/adamj/Downloads/myproject/images projet/cafarella-covid.jpg"
    image_path = "Image-covid.jpg"
    
    # Afficher l'image
    st.image(image_path, caption="Description de l'image", use_column_width=True)

elif selected == "Données":
    st.title("Données")
    st.write("""
    Les données dont nous disposons sont des images de radiographies issues de diverses sources académiques et 
    médicales. Elles sont constituées de 21165 observations. Pour chacune d’elle, les informations disponibles 
    consistent en :
    - Une image de radio au format png.
    - Chaque image est appairée à une image masque qui délimite la zone des poumons sur la radio. 
    - Un ensemble de métadonnées. Ces dernières indiquent notamment un label, à savoir la classe à laquelle 
      appartient chaque patient, et la source de la donnée.
    """)

    # Affichage du tableau des variables
    st.write("### Variables des métadonnées")
    variable_data = {
        "Variable": ["file_name", "format", "resolution", "url"],
        "Type": ["Object", "Object", "Object", "Object"],
        "Description": [
            "Nom de l’image indiquant le type de maladie et un numéro identifiant.",
            "Format de l’image. Toutes sont au format png.",
            "Résolution de l’image",
            "Lien url vers la source"
        ]
    }
    variable_df = pd.DataFrame(variable_data)
    st.table(variable_df)

elif selected == "Répartition par type de patient":
    st.title("Répartition par type de patient")
    st.write("""
    Les images radios sont labellisées selon le type de patients auxquelles elles correspondent. On distingue 4 classes 
    de données :
    - Les images Covid ;
    - Les images de patients atteints de pneumonie virale ;
    - Les images étiquetées lung opacity qui correspondent à des patients souffrant de diverses infections 
      pulmonaires ;
    - Les patients sains, labellisés normal.

    Comme illustré par la Figure 1, les données sont déséquilibrées. En effet, si les images radio sont équitablement 
    réparties entre patients sains (un peu plus de 10000, cf. Figure 1) et patients malades (environ 11000), les 
    malades consistent à 33% de patients covid, à 12% de malades de pneumonie virale et à 55% de malades atteints 
    de pneumonie bactérienne. Les patients covid, notre cible, représentent ainsi 17% des observations totales.
    Nous nous sommes donc assurés dans les phases de modélisation d’obtenir des classes équilibrées lors de 
    l’échantillonnage afin d’éviter d’introduire des biais dans nos modèles et d’assurer la précision de leurs 
    estimations.
    """)

    graph_choice = st.radio("", ("Distribution of Patient Types", "Distribution of Diseases", "Pie chart of Patient Types", "Pie chart of Diseases"))

    if graph_choice == "Distribution of Patient Types":
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Taille réduite
        data_meta['data_type'].value_counts().plot(kind='bar', color='darkcyan', edgecolor='black', ax=ax)
        plt.xticks(rotation=25, fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.title('Distribution of Patient Types', fontsize=10)
        st.pyplot(fig)
    elif graph_choice == "Distribution of Diseases":
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Taille réduite
        sick_count = data_meta.loc[data_meta.data_type != "Normal"].data_type.value_counts(normalize=True)
        sick_count.plot(kind='bar', color='darkseagreen', edgecolor='black', width=0.3, ax=ax)
        plt.xticks(rotation=25, fontsize=8)
        plt.ylabel('Percentage', fontsize=8)
        plt.title('Distribution of Diseases', fontsize=10)
        st.pyplot(fig)
    elif graph_choice == "Pie chart of Patient Types":
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Taille réduite
        data_meta['data_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.ylabel('')
        plt.title('Pie chart of Patient Types', fontsize=10)
        st.pyplot(fig)
    elif graph_choice == "Pie chart of Diseases":
        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Taille réduite
        sick_count = data_meta.loc[data_meta.data_type != "Normal"].data_type.value_counts(normalize=True)
        sick_count.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        plt.ylabel('')
        plt.title('Pie chart of Diseases', fontsize=10)
        st.pyplot(fig)

elif selected == "Répartition des sources":
    pass
    st.write("""
    Les données sont issues de 8 sources différentes provenant de dépôts kaggle et github, ainsi que de trois sites 
    de radiologie européens (cf Figure 2).
    Six des sources fournissent uniquement des images covid. Les images de patients sains proviennent des deux 
    dépôts kaggle rsna-pneumonia-detection-challenge et chest-xray-pneumonia, qui fournissent également 
    respectivement les images de pneumonies.
    """)

    # Définir la liste des labels simplifiés avant les graphiques
    simplified_labels = [
        "kaggle/rnsa", "kaggle/xray-pneum", "bimcv/covid19", "github/covid-cxnet", 
        "eurorad.org", "github/ml-workgroup", "github/covid-chestxray", "sirm/covid19"
    ]

    # Checklists pour sélectionner les graphiques à afficher
    options = st.multiselect("Choisissez les graphiques à afficher", ["Répartition des images par source", "Répartition des images par source et type de patient"])

    if "Répartition des images par source" in options:
        # Comptage des images par source
        url_counts = data_meta['url'].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))  # Ajustement de la taille de la figure
        ax.bar(url_counts.index, url_counts.values, color="mediumseagreen", edgecolor="black")
        ax.set_xticks(url_counts.index)
        ax.set_xticklabels(simplified_labels, rotation=42, fontsize=8)
        ax.set_ylabel('Effectifs', fontsize=10)
        ax.set_title('Répartition des images par source (Figure 2)', fontsize=12)
        st.pyplot(fig)

    if "Répartition des images par source et type de patient" in options:
        # Ajout du graphique Seaborn
        fig_seaborn, ax_seaborn = plt.subplots(figsize=(10, 6))
        bplot = sns.countplot(y='url', hue='data_type', data=data_meta, ax=ax_seaborn)
        ax_seaborn.set_yticks(range(len(simplified_labels)))
        ax_seaborn.set_yticklabels(simplified_labels, rotation=30)
        ax_seaborn.set_xlabel('Count', fontsize=10)
        ax_seaborn.set_ylabel('Source', fontsize=10)
        ax_seaborn.set_title('Répartition des images par source et type de patient', fontsize=12)
        st.pyplot(fig_seaborn)

    # Affichage du tableau croisé
    st.write("### Tableau croisé des images par source et type de patient")
    crosstab = pd.crosstab(data_meta.url, data_meta.data_type)
    st.dataframe(crosstab)

elif selected == "Formalisation du modèle":
    st.title("Formalisation du modèle de classification")
    st.header("Choix du nombre de classes")
    st.write("""
    1. Enjeu métier :
        - Identification précise des Covids ;
        - Distinction efficace des malades et sains ;
        - Recoupement potentiels entre les classes *Infections pulmonaires* et *Lung opacity*.
    2. Questions computationnelles :
        - Ressources en calculs ;
        - Temps d'entraînement des modèles.
    """)
    st.header("Formalisation")         
    st.write("Problème à trois classes équilibrées obtenues par échantillonnage :")
    st.markdown("""        
    * Les malades Covid ;
    * Les malades atteints de pathologies pulmonaires autres que le Covid (agrégation des classes 
      pneumonie virale et opacité pulmonaire) ;
    * Les patients sains.
     """)        
    st.header("Critères de performance retenus")
    st.write("**Critères globaux**")
    st.markdown("""          
    * Précision ;
    * f1-score.
     """)         
    st.write("**Sur la classe Covid**")        
    st.markdown("""         
    * Précision ;          
    * Rappel.
    """)

elif selected == "Modèles préselectionnés":
    st.title("Modèles présectionnés")
    st.write("""
    En phase exploratoire, test de multiples modèles, d'abord de Machine Learning, puis de Deep Learning. 
    """)

    show_option = st.radio("", ["Afficher la procédure", "Afficher les résultats", "Choix des modèles"], horizontal=True)

    if show_option == "Afficher la procédure":
        st.image("image_process_models.png", caption="Image Process Models")
    elif show_option == "Afficher les résultats":
        st.image("resultat_tab2.png", caption="Résultats")
    elif show_option == "Choix des modèles":

        st.markdown("**Sélection fondée sur les performances observées :**")
        st.markdown("- CNN qui obtient de bons résultats, notamment sur la classe Covid et qui est un modèle de Deep Learning simple à implémenter ;")
        st.markdown("- VGG16 dont les résultats sont parmi les meilleurs, mais qui est très coûteux en temps de calcul")

        st.markdown("**Présentation de ces deux modèles du point de vue de :**")
        st.markdown("1. Leurs performances en prédiction ;")
        st.markdown("2. Leur facilité d’implémentation et leur coût computationnel ;")
        st.markdown("3. Leur interprétabilité.")

elif selected == "Modélisation":
    st.title("Modélisation")

    # Code pour la modélisation CNN ou VGG
    st.write("Sélection du modèle à utiliser pour la prédiction :")
    model_choice = st.selectbox("Choisissez le modèle", ("CNN", "VGG"))

    if model_choice == 'CNN':
        st.image("architecture_cnn1.png", caption="Architecture du modèle CNN")
    elif model_choice == 'VGG':
        st.image("architecture_vgg.png", caption="Architecture du modèle VGG")

    path_data = data_folder  # "../../data"
    path_all_models = "../../models/"

    path_datasets = "../datasets"

    # path_list_files = os.path.join(path_datasets, 'val.csv')
    path_list_files = os.path.join(path_datasets, 'test.csv')

    if st.button("Load and Predict"):
        with st.spinner("Chargement du modèle et calcul des prédictions..."):
            _, val_acc_names, df_cm = pred_weights_img(model_choice,
                                               path_list_files,
                                               path_data=data_folder,
                                               path_all_models=path_all_models)

            # st.write(f"Accuracy: {val_acc_names}")
            st.markdown(f"<h2 style='color:green;'><b>Accuracy: {val_acc_names:.3f}</b></h2>", unsafe_allow_html=True)
            st.write(df_cm)

elif selected == "Démo":
    st.title("Prédiction et interprétation (Grad-CAM) d'une image de test")
    # Grad-CAM

    path_data = data_folder  # "../../data"
    path_all_models = "../../models/"

    path_datasets = "../datasets"
    path_datasets = os.path.join(path_datasets, 'test_gradcam.csv')
    df_data = pd.read_csv(path_datasets, index_col=0)

    params = {
        'CNN': {'new_size': (128, 128),
                'file_model': "CNN_weights.h5",
                'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
                'load_img': CNNload.load_img_mask,
                'model_specs': define_CNN_3layers_depth2,
                'last_conv_layer_name': "conv3",
                'make_gradcam_heatmap': gradcam.make_gradcam_heatmap_CNN},
        'VGG': {'new_size': (128, 128),
                'file_model': "VGG16_weights.h5",
                'class_map': {2: 'COVID', 0: 'Normal', 1: 'Non-COVID'},
                'load_img': VGGload.load_img_mask,
                'model_specs': define_VGG16model,
                'last_conv_layer_name': "block5_conv3",
                'make_gradcam_heatmap': gradcam.make_gradcam_heatmap_VGG16},
    }

    st.write("Sélection du modèle à utiliser pour la prédiction :")
    model_choice = st.selectbox("Choisissez le modèle", ("CNN", "VGG"))

    str_im = st.selectbox("Sur quelle image tester le modèle ?", ("", "0", "1", "2", "3", "4", "5"))
    idx_im = -1 if not str_im else int(str_im)
    # st.write("numéro de l'image:", idx_im)

    if idx_im >= 0:
        row_data = df_data.iloc[idx_im, :]
        scan_path = os.path.join(path_data,
                                 row_data['img_folder'],
                                 row_data['Images'])
        mask_path = os.path.join(path_data,
                                 row_data['mask_folder'],
                                 row_data['Masks'])
        img_path = (scan_path, mask_path)
        st.write(f"-----------Image sélectionnée : {idx_im}-----------")

        # The true class
        class_map = params[model_choice]['class_map']
        true_cl_name = class_map[row_data['num_class']]
        st.write("Classe réelle :", true_cl_name)

        # display using Streamlit
        st.image(img_path[0], caption="Original scan")

        # Load image into array of appropriate shape for the model.
        img_size = params[model_choice]['new_size']
        img_array = params[model_choice]['load_img'](img_path,
                                                     new_size=img_size)

        # Make model.
        path_model = os.path.join(path_all_models,
                                  params[model_choice]['file_model'])
        model = params[model_choice]['model_specs'](*img_size,
                                                    num_classes=3,
                                                    summary=True)
        model.load_weights(path_model)
        model.layers[-1].activation = None
        st.write(f"-----------Modèle sélectionné : {model_choice} -----------")

        # Generate class activation heatmap
        last_conv_layer_name = params[model_choice]['last_conv_layer_name']
        heatmap_function = params[model_choice]['make_gradcam_heatmap']
        heatmap = heatmap_function(img_array, model, last_conv_layer_name)

        # Display results in st.image
        gradcam.save_and_display_gradcam(img_path, heatmap['heatmap'])

        # The predicted class
        st.write("Classe prédite :", class_map[heatmap["class_id"]])

elif selected == "Conclusion et perspectives":
    st.title("Conclusion et perspectives")
    st.markdown("**Proposition de deux modélisations ayant chacune des points forts différents :**")
    st.markdown("- CNN : modèle flexible, facile à implémenter et peu coûteux en ressources ;")
    st.markdown("- VGG : modèle de transfer learning obtenant une meilleure précision et un bon équilibre dans les performances, mais coûteux en calculs.")

    st.markdown("**Perspectives et pistes d'amélioration :**")
    st.markdown("- Affiner la calibration des hyperparamètres des modèles ;")
    st.markdown("- Ameliorer leur robustesse ;")
    st.markdown("- Se confronter à une expertise médicale pour évaluer les résultats obtenus en termes d'interprétabilité.")
