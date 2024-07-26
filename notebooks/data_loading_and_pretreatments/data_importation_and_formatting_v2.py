import os
import pandas as pd
import numpy as np
import cv2
import random
from collections import Counter

from data_loading_and_pretreatments.outlier_identification import outliers_id

data_folder = "../data"
conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]

##Fixe la seed pour le tirage aléatoire des images
random.seed(58)

## Fixe la résolution d'image sélectionnée
def set_img_size(size):
    global IMG_SIZE
    IMG_SIZE = size

def mask_overlay(img, mask):
    """
    Redimmensionne le masque et l'image à la taille IMG_SIZE puis superpose des deux
    """ 
    global IMG_SIZE
    mask = cv2.resize(mask, dsize=(IMG_SIZE,IMG_SIZE))
    img = cv2.resize(img, dsize=(IMG_SIZE,IMG_SIZE))
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def convert_to_vector(img):
    """
    Convertit une image au format array IMG_SIZE*IMG_SIZE en un vecteur.
    """ 
    global IMG_SIZE
    return img.reshape(1,IMG_SIZE*IMG_SIZE)[0]


def load_img_dir(condition, size=0, mask=False):
    """
    Charge un nombre size d'images pour une condition donnée après avoir écarté les outliers. 

    Args :
        condition : un string indiquant la condition à charger ;
        size : int indiquant le nombre d'images à charger. Si size=0, l'ensemble des images est chargé ;
        mask : un booléen indiquant si on charge les images radio ou les images de masques.

    Retourne un tuple contenant un array d'images (arrays 299 x 299) et une liste de labels
    """
    if mask==False:
        dir_condition = os.path.join(data_folder, condition, "images")
    else:
        dir_condition = os.path.join(data_folder, condition, "masks")  

    all_filenames = [name for name in os.listdir(dir_condition)]
    # Retrait des outliers     
    list_outliers = outliers_id()
    filenames = list((Counter(all_filenames) - Counter(list_outliers)).elements())

    if size!=0 :
        filenames = random.sample(filenames, size)
    img_list = []
    for i, filename in enumerate(filenames):
        img_list.append(cv2.imread(os.path.join(dir_condition, filename), cv2.IMREAD_GRAYSCALE))

    labels = np.repeat(condition, len(filenames))
    return np.array(img_list), labels




def load_img_dir_in_df(condition, size=0, mask=False):

    """
    Charge un nombre size d'images pour une condition donnée après avoir écarté les outliers. 

    Args :
        condition : un string indiquant la condition à charger ;
        size : int indiquant le nombre d'images à charger. Si size=0, l'ensemble des images est chargé ;
        mask : un booléen indiquant si on charge les images radio ou les images de masques.

    Retourne un df de IMG_SIZExIMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """   
    if mask==False:
        dir_condition = os.path.join(data_folder, condition, "images")
    else:
        dir_condition = os.path.join(data_folder, condition, "masks")  

    all_filenames = [name for name in os.listdir(dir_condition)]
    # Retrait des outliers     
    list_outliers = outliers_id()
    filenames = list((Counter(all_filenames) - Counter(list_outliers)).elements())

    if size!=0 :
        filenames = random.sample(filenames, size)
    img_list = []
    for i, filename in enumerate(filenames):
        img_list.append(cv2.imread(os.path.join(dir_condition, filename), cv2.IMREAD_GRAYSCALE))
    img_list = pd.Series(img_list)
    img_df = pd.DataFrame(img_list.apply(convert_to_vector).apply(pd.Series))
    img_df['label'] = condition
    return img_df


def load_img_multiple_cond_in_df(selected_conditions = 'all', sample_sizes=0): 
    """
    Charge un nombre fixé d'images radio pour une liste de conditions donnée après avoir écarté les outliers. 

    Args :
        selected_conditions : liste de conditions à charger. Ensemble de la base chargée si selected_conditions='all'
        sample_sizes : Liste de tailles d'échantillon pour les conditions renseignées. Si égal à 0, l'ensemble des images de la condition est chargé.

    Retourne un df de IMG_SIZE*IMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """  
    conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]
    if selected_conditions=='all': selected_conditions=conditions

    if (sample_sizes!=0) & (type(sample_sizes)=='int'):
        print("Les tailles d'échantillons ne correspondent pas au nombre de conditions")
    elif sample_sizes == 0:
        sample_sizes = np.zeros(len(selected_conditions))

    if (np.mean([c in conditions for c in selected_conditions])!=1)&(selected_conditions!=conditions): 
        print('Conditions incorrectes')
        res = None
    elif type(selected_conditions)==str:    # si un seul élément renseigné
        res = load_img_dir_in_df(selected_conditions, sample_sizes[0])
        res['label'] = selected_conditions
    else: 
        res = load_img_dir_in_df(selected_conditions[0], sample_sizes[0])
        res['label'] = selected_conditions[0]

        for c in range(1,len(selected_conditions)):
            img_df_c = load_img_dir_in_df(selected_conditions[c], sample_sizes[c])
            img_df_c['label'] = selected_conditions[c]
            res = pd.concat([res,img_df_c])
        res = res.reset_index().drop(columns='index')
    return res


def load_masked_img_dir(condition, size=0): 
    """
    Charge un nombre size d'images masquées pour une condition donnée après avoir écarté les outliers. 

    Args :
        condition : un string indiquant la condition à charger
        size : int indiquant le nombre d'images à charger. Si size=0, l'ensemble des images est chargé.
    Retourne un tuble contenant un array d'images (arrays IMG_SIZExIMG_SIZEx1) et 1 colonne de labels
    """     
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    all_filenames = [name for name in os.listdir(dir_radio_images)]

    # Retrait des outliers     
    list_outliers = outliers_id()
    filenames = list((Counter(all_filenames) - Counter(list_outliers)).elements())
    
    if size!=0 :
        size = np.min([size, len(filenames)])
        filenames = random.sample(filenames, size)
    masked_img_list = []
    
    for i, filename in enumerate(filenames):
        # img = load_bw_img(os.path.join(dir_condition, "images", filename))
        img = cv2.imread(os.path.join(dir_condition, "images", filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(dir_condition, "masks", filename), cv2.IMREAD_GRAYSCALE)
        masked_img_list.append(mask_overlay(img,mask)[:, :, np.newaxis])

    labels = np.repeat(condition, len(filenames))
    return np.array(masked_img_list), labels


def load_masked_img_multiple_cond(selected_conditions = 'all', sample_sizes=0):
    """
    Charge un nombre fixé d'images radio masquées pour une liste de conditions donnée après avoir écarté les outliers. 

    Args :
        selected_conditions : liste de conditions à charger. Ensemble de la base chargée si selected_conditions='all'
        sample_sizes : Liste de tailles d'échantillon pour les conditions renseignées. Si égal à 0, l'ensemble des images de la condition est chargé.

    Retourne un tuble contenant un array d'images (arrays IMG_SIZExIMG_SIZEx1) et 1 colonne de labels
    """  
    conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]

    if selected_conditions=='all': selected_conditions=conditions

    if (sample_sizes!=0) & (type(sample_sizes)=='int'):
        print("Les tailles d'échantillons ne correspondent pas au nombre de conditions")
    elif sample_sizes == 0:
        sample_sizes = np.zeros(len(selected_conditions))

    if (np.mean([c in conditions for c in selected_conditions])!=1)&(selected_conditions!=conditions): 
        print('Conditions incorrectes')
        res = None

    elif type(selected_conditions)==str:    # si un seul élément renseigné
        res, labels = load_masked_img_dir(selected_conditions, sample_sizes[0])
    else: 
        res, labels = load_masked_img_dir(selected_conditions[0], sample_sizes[0])

        for c in range(1,len(selected_conditions)):
            img_cond, labels_cond = load_masked_img_dir(selected_conditions[c], sample_sizes[c])
            res = np.concatenate([res,img_cond])
            labels = [*labels, *labels_cond]
    return res, labels

def load_masked_img_dir_in_df(condition, size=0):
    """
    Charge un nombre size d'images masquées pour une condition donnée après avoir écarté les outliers. 

    Args :
        condition : un string indiquant la condition à charger
        size : int indiquant le nombre d'images à charger. Si size=0, l'ensemble des images est chargé.
    Retourne un df de IMG_SIZExIMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """     
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    all_filenames = [name for name in os.listdir(dir_radio_images)]

    # Retrait des outliers     
    list_outliers = outliers_id()
    filenames = list((Counter(all_filenames) - Counter(list_outliers)).elements())

    if size!=0 :
        size = np.min([size, len(filenames)])
        filenames = random.sample(filenames, size)
    masked_img_list = []
    for i, filename in enumerate(filenames):
        # img = load_bw_img(os.path.join(dir_condition, "images", filename))
        img = cv2.imread(os.path.join(dir_condition, "images", filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(dir_condition, "masks", filename), cv2.IMREAD_GRAYSCALE)
        masked_img_list.append(mask_overlay(img,mask))
    masked_img_list = pd.Series(masked_img_list)
    masked_img_df = pd.DataFrame(masked_img_list.apply(convert_to_vector).apply(pd.Series))
    masked_img_df['label'] = condition
    return masked_img_df


def load_masked_img_multiple_cond_in_df(selected_conditions = 'all', sample_sizes=0):
    """
    Charge un nombre fixé d'images radio masquées pour une liste de conditions donnée après avoir écarté les outliers. 
    
    Args :
        selected_conditions : liste de conditions à charger. Ensemble de la base chargée si selected_conditions='all'
        sample_sizes : Liste de tailles d'échantillon pour les conditions renseignées. Si égal à 0, l'ensemble des images de la condition est chargé.

    Retourne un df de IMG_SIZE*IMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """  
    conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]

    if selected_conditions=='all': selected_conditions=conditions

    if (sample_sizes!=0) & (type(sample_sizes)=='int'):
        print("Les tailles d'échantillons ne correspondent pas au nombre de conditions")
    elif sample_sizes == 0:
        sample_sizes = np.zeros(len(selected_conditions))

    if (np.mean([c in conditions for c in selected_conditions])!=1)&(selected_conditions!=conditions): 
        print('Conditions incorrectes')
        res = None

    elif type(selected_conditions)==str:    # si un seul élément renseigné
        res = load_masked_img_dir_in_df(selected_conditions, sample_sizes[0])
        res['label'] = selected_conditions
    else: 
        res = load_masked_img_dir_in_df(selected_conditions[0], sample_sizes[0])
        res['label'] = selected_conditions[0]

        for c in range(1,len(selected_conditions)):
            img_df_c = load_masked_img_dir_in_df(selected_conditions[c], sample_sizes[c])
            img_df_c['label'] = selected_conditions[c]
            res = pd.concat([res,img_df_c])
        res = res.reset_index().drop(columns='index')
    return res
