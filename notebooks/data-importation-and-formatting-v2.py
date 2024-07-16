import os
import pandas as pd
import numpy as np
import cv2
import random

data_folder = "../data"
conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]
img_folder = "images"

lst_dirs_condition = [os.path.join(data_folder, cond, "images") for cond in conditions]

list_dir_images = pd.DataFrame({"img_type" : ["Viral Pneumonia", "Bacterial Pneumonia", "Covid", "Normal"],
                                 "img_dir" : lst_dirs_condition})
                                 
IMG_SIZE = 50

def mask_overlay(img, mask):
    """
    Redimmensionne le masque et l'image à la taille IMG_SIZE puis superpose des deux
    """ 
    mask = cv2.resize(mask, dsize=(IMG_SIZE,IMG_SIZE))
    img = cv2.resize(img, dsize=(IMG_SIZE,IMG_SIZE))
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def convert_to_vector(img):
    """
    Convertit une image au format array IMG_SIZE*IMG_SIZE en un vecteur.
    """ 
    return img.reshape(1,IMG_SIZE*IMG_SIZE)[0]


def load_img_dir(condition, mask=False):
    """
    Charge l'ensemble des images radio ou masques pour une condition donnée.
    Chaque image est chargée au format array IMG_SIZE*IMG_SIZE.
    Retourne une liste d'arrays.
    """     
    if mask==False:
        dir_condition = os.path.join(data_folder, condition, "images")
    else:
        dir_condition = os.path.join(data_folder, condition, "masks")    
    filenames = [name for name in os.listdir(dir_condition)]
    img_list = []
    for i, filename in enumerate(filenames):
        img_list.append(cv2.imread(os.path.join(dir_condition, filename), cv2.IMREAD_GRAYSCALE))
    return img_list

def load_img_dir_in_df(condition, mask=False):
    """
    Charge l'ensemble des images radio ou masques pour une condition donnée.
    Retourne un df de IMG_SIZExIMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """     
    if mask==False:
        dir_condition = os.path.join(data_folder, condition, "images")
    else:
        dir_condition = os.path.join(data_folder, condition, "masks")       
    filenames = [name for name in os.listdir(dir_condition)]
    img_list = []
    for i, filename in enumerate(filenames):
        img_list.append(cv2.imread(os.path.join(dir_condition, filename), cv2.IMREAD_GRAYSCALE))
    img_list = pd.Series(img_list)
    img_df = pd.DataFrame(img_list.apply(convert_to_vector).apply(pd.Series))
    img_df['label'] = condition
    return img_df

def load_img_multiple_cond_in_df(selected_conditions = 'all'):
    """
    Charge l'ensemble des images radio ou masques pour une liste de conditions donnée donnée.
    Retourne un df de IMG_SIZE*IMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """  
    conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]
    if selected_conditions=='all': selected_conditions=conditions
    if (np.mean([c in conditions for c in selected_conditions])!=1)&(selected_conditions!=conditions): 
        print('Conditions incorrectes')
        res = None
    elif type(selected_conditions)==str:    # si un seul élément renseigné
        res = load_img_dir_in_df(selected_conditions)
        res['label'] = selected_conditions
    else: 
        res = load_img_dir_in_df(selected_conditions[0])
        res['label'] = selected_conditions[0]

        for c in range(1,len(selected_conditions)):
            img_df_c = load_img_dir_in_df(selected_conditions[c])
            img_df_c['label'] = selected_conditions[c]
            res = pd.concat([res,img_df_c])
        res = res.reset_index().drop(columns='index')
    return res


def load_masked_img_dir(condition, size=None):
    """
    Charge un nombre size d'images radio ou de masques pour une condition donnée.
    Chaque image est chargée au format array IMG_SIZE*IMG_SIZE.
    Retourne une liste d'arrays.
    """    
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    filenames = [name for name in os.listdir(dir_radio_images)]
    if size!=None :
        filenames = random.sample(filenames, size)
    masked_img_list = []
    for i, filename in enumerate(filenames):
        # img = load_bw_img(os.path.join(dir_condition, "images", filename))
        img = cv2.imread(os.path.join(dir_condition, "images", filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(dir_condition, "masks", filename), cv2.IMREAD_GRAYSCALE)
        masked_img_list.append(mask_overlay(img,mask))
    return masked_img_list

def load_masked_img_dir_in_df(condition, size=0):
    """
    Charge un nombre size d'images masquées pour une condition donnée. 
    Args :
        condition : un string indiquant la condition à charger
        size : int indiquant le nombre d'images à charger. Si size=0, l'ensemble des images est chargé.
    Retourne un df de IMG_SIZExIMG_SIZE colonnes (1 par pixel) + 1 colonne de label
    """     
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    filenames = [name for name in os.listdir(dir_radio_images)]
    if size!=0 :
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
    Charge un nombre fixé d'images radio masquées pour une liste de conditions donnée. 
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
