import os
import pandas as pd
import numpy as np
import cv2

data_folder = "../data"
conditions = ["Viral Pneumonia", "Lung_Opacity", "COVID",  "Normal"]
img_folder = "images"

lst_dirs_condition = [os.path.join(data_folder, cond, "images") for cond in conditions]

list_dir_images = pd.DataFrame({"img_type" : ["Viral Pneumonia", "Bacterial Pneumonia", "Covid", "Normal"],
                                 "img_dir" : lst_dirs_condition})

def mask_overlay(img, mask):
    mask = cv2.resize(mask, dsize=img.shape)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def convert_to_vector(img):
    return img.reshape(1,299*299)[0]

def load_img_dir(condition, mask=False):
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

def get_all_masks_size():
    condition = conditions[0]
    dir_condition = os.path.join(data_folder, condition, "masks") 
    filenames = [name for name in os.listdir(dir_condition)]
    part_lungs_size = []
    for i, filename in enumerate(filenames):
        mask = cv2.imread(os.path.join(dir_condition, filename), cv2.IMREAD_GRAYSCALE)
        part_lungs_size.append(np.mean(mask==255))
    df_mean_lungs_size = pd.DataFrame(part_lungs_size, columns=['lung_portion'])
    df_mean_lungs_size['label'] = condition

    for c in range(1,len(conditions)):
        condition = conditions[c] 
        dir_condition = os.path.join(data_folder, condition, "masks") 
        filenames = [name for name in os.listdir(dir_condition)]
        part_lungs_size = []
        for i, filename in enumerate(filenames):
            mask = cv2.imread(os.path.join(dir_condition, filename), cv2.IMREAD_GRAYSCALE)
            part_lungs_size.append(np.mean(mask==255))
        part_lungs_size = pd.DataFrame(part_lungs_size, columns=['lung_portion'])
        part_lungs_size['label'] = condition
        df_mean_lungs_size = pd.concat([df_mean_lungs_size,part_lungs_size], axis=0)
    return df_mean_lungs_size

def load_masked_img_dir(condition):
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    filenames = [name for name in os.listdir(dir_radio_images)]
    masked_img_list = []
    for i, filename in enumerate(filenames):
        # img = load_bw_img(os.path.join(dir_condition, "images", filename))
        img = cv2.imread(os.path.join(dir_condition, "images", filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(dir_condition, "masks", filename), cv2.IMREAD_GRAYSCALE)
        masked_img_list.append(mask_overlay(img,mask))
    return masked_img_list

def load_masked_img_dir_in_df(condition):
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    filenames = [name for name in os.listdir(dir_radio_images)]
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

def get_color_distribution(masked_img_asserie):
    # Retrait du noir
    masked_im = masked_img_asserie[masked_img_asserie!=0]
    # Calcul de la répartition des niveaux de gris
    col_distr = pd.DataFrame(masked_im).value_counts(normalize=True).sort_index()
    pixel_values = col_distr.index
    col_distr = pd.DataFrame(col_distr.values).transpose()
    col_distr.columns = [n[0] for n in list(pixel_values)] 

    # Remplissage du tableau pour obtenir une distribution sur l'ensemble de définition [1,255]
    # Les proportions de pixels non présents seront 0.
    null_distrib = pd.DataFrame(np.zeros(255),index=np.arange(1,256)).transpose()
    df_distrib = pd.concat([null_distrib, col_distr], axis=0)
    return  df_distrib.tail(1).fillna(0)

def compute_color_distribution_from_dir_imgs(condition):
    dir_condition = os.path.join(data_folder, condition)
    dir_radio_images = os.path.join(dir_condition, "images")
    filenames = [name for name in os.listdir(dir_radio_images)]
    colors_distrib_dir =  pd.DataFrame(np.zeros(255),index=np.arange(1,256)).transpose()
    for i, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(dir_condition, "images", filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(dir_condition, "masks", filename), cv2.IMREAD_GRAYSCALE)
        masked_img = mask_overlay(img,mask)
        img_col_distr =  get_color_distribution(masked_img)
        colors_distrib_dir = pd.concat([colors_distrib_dir, img_col_distr], axis=0)
    return colors_distrib_dir.tail(-1).reset_index().drop(columns='index')