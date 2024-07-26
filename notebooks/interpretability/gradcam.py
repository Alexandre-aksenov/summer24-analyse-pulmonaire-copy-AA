##

import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from keras.utils import  load_img
from tensorflow.keras import layers, Input
from tensorflow.keras.models import  Model
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt

data_folder = "../data"


def make_gradcam_heatmap(
    img_array, model, 
    last_conv_layer_name, 
    classifier_layer_names,
    class_indices
):
    """
    Calcule la heatmap GradCam associée à chacune des classes pour une image donnée.
    Args :
        img_array : une image au format np.array ;
        model : le modèle étudié ;
        last_conv_layer_name : le nom de la couche de convolution utilisée pour évaluer le gradient ;
        classifier_layer_names : la liste des noms des couches de classification (toutes les couches qui suivent l'extraction de features) ;
        class_indices : la liste des classes
    Returns :
        Une liste de dictionnaires comportant deux éléments : l'identifiant de la classe et un array correspondant à la heatmap gradcam associée,
        i.e {"class_id", "heatmap"}

    Cette fonction est une adaptation de la fonction du même nom du notebook https://www.kaggle.com/code/gowrishankarin/gradcam-model-interpretability-vgg16-xception/notebook
    """
    #1. Create a model that maps the input image to the activations of the last convolution layer - Get last conv layer's output dimensions
    last_conv_layer = model.get_layer('vgg16').get_layer(last_conv_layer_name)
    last_conv_layer_model = Model(model.get_layer('vgg16').inputs, last_conv_layer.output)
    
    #2. Create another model, that maps from last convolution layer to the final class predictions - This is the classifier model that calculated the gradient
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = Model(classifier_input, x)
    
    n_classes = len(class_indices)

    #3. Create an array to store the heatmaps
    heatmaps = []
    #4. Iteratively calculate heatmaps for all classes of interest using GradientTape
    for index in np.arange(n_classes):
    
        #5. Watch the last convolution output during the prediction process to calculate the gradients
        #6. Compute the activations of last conv layer and make the tape to watch
        with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)

            #7. Get the class predictions and the class channel using the class index
            preds = classifier_model(last_conv_layer_output)
            class_channel = preds[:, class_indices[index]]
            
        #8. Using tape, Get the gradient for the predicted class wrt the output feature map of last conv layer    
        grads = tape.gradient(
            class_channel,
            last_conv_layer_output
        )
        
        #9. Calculate the mean intensity of the gradient over its feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        #10. Multiply each channel in feature map array by weight importance of the channel
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        #11. The channel-wise mean of the resulting feature map is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        #12. Normalize the heatmap between [0, 1] for ease of visualization
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        heatmaps.append({
            "class_id": class_indices[index],
            "heatmap": heatmap
        })

    return heatmaps


def save_and_display_gradcam(img_array, img_name, heatmap, alpha=0.4):
    """
    Superpose et sauvegarde l'image originale et la heatmpa GradCam associée.
    Args :
        img_array : une image au format np.array ;
        img_name : le nom de l'image sous lequel elle sera enregistrée ;
        heatmap : la heatmap gradcam pour l'image img_array, au format array numpy ;
        alpha : paramètre graphique.
    Returns :
        L'image de superposition au format array numpy.

    Cette fonction est une adaptation de la fonction du même nom fournie par la documentation Keras https://keras.io/examples/vision/grad_cam/. 
    """
    # Save and reload the original image
    img_name = img_name+".png"
    img_path = os.path.join(data_folder, 'test_gradcam', img_name)
    array_to_img(img_array).save(img_path)
    
    img = load_img(img_path)
    img = img_to_array(img)
    # img = np.array(img, np.float32)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img =  jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    cam_path =  os.path.join(data_folder, 'test_gradcam', "grad"+img_name+".png")

    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
    return superimposed_img
