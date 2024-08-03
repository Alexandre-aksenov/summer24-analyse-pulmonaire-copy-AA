import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
# Display
from IPython.display import Image, display
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras import Input
from tensorflow.keras.models import  Model


def make_gradcam_heatmap_CNN(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    # https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # return heatmap.numpy()
    return {'class_id': int(pred_index), 'heatmap' : heatmap.numpy()}


def make_gradcam_heatmap_VGG16(
    img_array, model,
    last_conv_layer_name,
    pred_index=None
):
    """
    Calcule la heatmap GradCam associée à chacune des classes pour une image donnée.
    Args :
        img_array : une image au format np.array ;
        model : le modèle étudié ;
        last_conv_layer_name : le nom de la couche de convolution utilisée pour évaluer le gradient ;
    Returns :
        Une liste de dictionnaires comportant deux éléments : l'identifiant de la classe et un array correspondant à la heatmap gradcam associée,
        i.e {"class_id", "heatmap"}

    Cette fonction est une adaptation de la fonction du même nom du notebook https://www.kaggle.com/code/gowrishankarin/gradcam-model-interpretability-vgg16-xception/notebook
    """
    
    # Liste des noms des couches de classification (toutes les couches qui suivent l'extraction de features) 
    classifier_layer_names = [layer.name for layer in model.layers][1:]
    
    # La liste des classes
    class_indices = [0,1,2]

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

    res = None
    if pred_index == None :
        res = heatmaps[class_indices[np.array(preds[0]).argmax()]]
    else :
        res = heatmaps[pred_index]

    return res


def save_and_display_gradcam(img_path, heatmap, cam_path="cam_v0.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path[0])
    img = keras.utils.img_to_array(img)

    # cam_path = "cam_img_" + str(idx_im) + ".jpg"
    cam_path = "cam_img_" + os.path.split(img_path[0])[1]

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    # jet = mpl.colormaps["jet"]
    jet = cm.jet

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    display(superimposed_img)