import os
import pandas as pd
import numpy as np
import cv2

data_folder = "../data"
conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]
lst_dirs_condition = [os.path.join(data_folder, cond, "images") for cond in conditions]
list_dir_images = pd.DataFrame({"img_type" : ["Viral Pneumonia", "Lung Opacity", "Covid", "Normal"],
                            "img_dir" : lst_dirs_condition})

def color_concentration(img, alpha):
    """
   Arg : 
        img : np.array en 1 dim
        alpha : niveau de proba 
   Returns :
        Indicateur (scalaire) : écart interquantile (d'amplitude 1-alpha/2)
    """
    i1 = (np.diff(np.quantile(img, q=(0+alpha/2,1-alpha/2))))#/(np.max(img) - np.min(img))
    return i1[0]

def contrast_img_indicators(alpha):
    '''
    Pour chacune des images radio et pour chaque condition, calcule la moyenne et l'écart-type des niveaux de gris.
    
    Return
        Un dataframe contenant 4 colonnes : le nom de l'image, son niveau de gris moyen, l'écart-type et l'écart interquantile des niveaux de gris.
    '''   
    if os.path.exists(os.path.join(data_folder, "img_contrasts_indicators.csv")):
        contrast_df = pd.read_csv(os.path.join(data_folder, "img_contrasts_indicators.csv"), index_col=0)
    else :
        contrast_df = []
        for k, dir in enumerate(list_dir_images.img_dir):
            dir_images = dir
            filenames = [name for name in os.listdir(dir_images)]
            files_size = len(filenames)
            for i, filename in enumerate(filenames):
                curr_img = cv2.imread(os.path.join(dir_images, filename))
                contrast_df.append([filename.split('.')[0], np.mean(curr_img), np.std(curr_img), color_concentration(curr_img, alpha)])


        contrast_df = pd.DataFrame(contrast_df, columns=["file_name", "color_mean", "color_std", "color_concentration"])
        contrast_df.to_csv(os.path.join(data_folder, "img_contrasts_indicators.csv"))
    return contrast_df


def outliers_id(level=0.005, alpha=0.05):
    """
    Identifie les images les p% les moins contrastées (dans le sens des niveaux de dipersion des gris)

    Args :
        level : le niveau p de probabilité en deçà duquel les images sont supposées aberrantes.

    Return
        Une liste de noms d'images
    """   
    if os.path.exists(os.path.join(data_folder, "img_contrasts_indicators.csv")):
        contrast_img_data = pd.read_csv(os.path.join(data_folder, "img_contrasts_indicators.csv"), index_col=0)
    else :
        contrast_img_data = contrast_img_indicators(alpha)
    seuils_outliers_contrast = np.quantile(contrast_img_data.color_std, q=(level))
    low_contrast = contrast_img_data[(contrast_img_data.color_std<seuils_outliers_contrast)]
    low_contrast_img_names = low_contrast.file_name + '.png'
    return low_contrast_img_names

def get_all_masks_size():
    """
    Calcule la proportion de chaque image occupée par les poumons en évaluant la part de blanc dans chaque masque.

    Return
        Un dataframe contenant 3 colonnes : le nom de l'image, sa classe et la part occupée par les poumons.
    """   
    lst_masks_dirs_condition = [os.path.join(data_folder, cond, "masks") for cond in conditions]
    list_dir_masks = pd.DataFrame({"img_type" : ["Viral Pneumonia", "Lung Opacity", "Covid", "Normal"],
                                "masks_dir" : lst_masks_dirs_condition})

    part_lungs_size = []
    for k, dir in enumerate(list_dir_masks.masks_dir):
        filenames = [name for name in os.listdir(dir)]
        files_size = len(filenames)
        for i, filename in enumerate(filenames):
            curr_mask = cv2.imread(os.path.join(dir, filename))
            part_lungs_size.append([filename, filename.split('-')[0], np.mean(curr_mask==255)])

    df_mean_lungs_size = pd.DataFrame(part_lungs_size, columns=["file_name", "label", "lung_portion"])
    return df_mean_lungs_size

def get_black_proportion_in_img():
    """
    Calcule la proportion de noir dans chaque image 

    Return
        Un dataframe contenant 3 colonnes : le nom de l'image, sa classe et la part de pixels noirs.
    """   
    prop_black_img = []
    for k, dir in enumerate(list_dir_images.img_dir):
        dir_images = dir
        filenames = [name for name in os.listdir(dir_images)]
        files_size = len(filenames)
        for i, filename in enumerate(filenames):
            curr_img = cv2.imread(os.path.join(dir_images, filename))
            prop_black_img.append([filename, filename.split('-')[0], np.mean(curr_img==0)])

    df_prop_black_img = pd.DataFrame(prop_black_img, columns=["file_name", "label", "black_prop"])
    return df_prop_black_img