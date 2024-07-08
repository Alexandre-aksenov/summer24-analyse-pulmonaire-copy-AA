""" 
Functions for loading images. Extracted from the script of PCA.

For use in V5.

Accelerate image loading by preallocating, then filling the tensor, TODO.
"""

import os
import pandas as pd
import cv2

import numpy as np

def fname_num(fname: str):
    """
    'Type'_'num'.png
    (str)
    ->
    num (integer)
    
    Example:
    >>> fname_num("Viral Pneumonia-42.png")
    42
    """
    str_num = fname.split('.')[0].split('-')[-1]
    return int(str_num)


def path2DF_iter_im_cv2(path_folder):
    """
    path to folder with images
    ->

    iterator of tuples:
    image numbers, images (matrices or tensors)

    in desorder.
    """
    file_names = os.listdir(path=path_folder)
    # iter_images = ((fname_num(fname), mpimg.imread(os.path.join(path_folder, fname))) for fname in file_names)
    # +>
    iter_images = ((fname_num(fname), cv2.imread(os.path.join(path_folder, fname), cv2.IMREAD_GRAYSCALE)) for fname in file_names)    
    
    # -> cv2.imread
    return iter_images

def extract_feat_2DF_opt_cols(fun_feat, iter_img, feat_names=None):
    """
    Input.
    
    fun_feat : matrix with an image -> row vector of features;
    iter_img (iterable):
            iterator of tuples:
            image numbers, images (matrices or tensors) ;
    feat_names (list-like):  feature names, sent to the constructor of pd.DataFrame. Default=None.
    
    Output.
    DF (N * features) with:
    image number as index, 
    features as data
    """

    lst_num_shape = [(id_mat[0], fun_feat(id_mat[1])) for id_mat in iter_img]
    # list of (index, arrays of features)
    
    # Conversion dict -> sorted DF
    dict_num_shape = dict(lst_num_shape)
    df_shapes = pd.DataFrame.from_dict(dict_num_shape, orient='index', columns=feat_names)
    df_shapes = df_shapes.sort_index()
    return df_shapes



def flatten_except_1st_dim(tens):
    """

    Args:
        tens (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tens.reshape(tens.shape[0], -1)


def path2DF_imgs(path_folder):
    """path2DF_imgs _summary_

    Args:
        path_folder (_type_): _description_

    Returns:
        _type_: _description_
    """
    # num_files = len(os.listdir(path=path_folder))
    df_sizes = extract_feat_2DF_opt_cols(lambda mat: (mat.shape),
                            path2DF_iter_im_cv2(path_folder))
    
    # extact sizes 
    img_height = df_sizes.iloc[0, 0]
    img_width = df_sizes.iloc[0, 1]
    num_img = df_sizes.shape[0]
    
    tens_img = np.zeros((num_img, img_height, img_width), dtype=np.uint8)
    
    # fill the tensor
    for (im_num, image) in path2DF_iter_im_cv2(path_folder):
        tens_img[im_num - 1, :, :] = image
    
    # convert to DF
    df_images = pd.DataFrame(data=flatten_except_1st_dim(tens_img),
                            index=range(1, num_img + 1))
    return df_images


def load_list_paths(lst_paths: list, var_target: str):
    """
    Should be applied to the list of paths to conditions:
    ['COVID', 'Normal', 'Lung Opacity', 'Viral Pneumonia'].

    Args:
        lst_paths (list[str]): list of paths to folders with conditions.
        var_target (str)
    Returns:
        DataFrame : data for classification.
        

    """
    lst_DFs = list(map(path2DF_imgs, lst_paths))
    for num_cond, DF_cond in enumerate(lst_DFs):
        DF_cond[var_target] = num_cond * np.ones((DF_cond.shape[0],), dtype=np.uint8)
    
    df = pd.concat(lst_DFs, axis=0)
    # set index to avoid repetitions
    df.set_index(np.arange(df.shape[0]), inplace=True)
    
    return df
