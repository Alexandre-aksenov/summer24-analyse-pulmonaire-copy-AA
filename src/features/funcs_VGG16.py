"""
Functions for defining a VGG16 pretrained model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from timeit import default_timer as timer
from tensorflow.keras.applications import VGG16
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Callbacks
class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


def pred_mat2class(pred_matrix):
    return pred_matrix.argmax(axis=1)


def define_VGG16model(IMG_WIDTH,
                              IMG_HEIGHT,
                              num_classes,
                              n_retrained_vgg_layers = 4,
                              summary=True):
    """
    Defines and compiles a VGG16 model.
    """
    pre_trained_model = VGG16(weights = 'imagenet', include_top=False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    for layer in pre_trained_model.layers:
        layer.trainable = False
    for layer in pre_trained_model.layers[-4:]:
        layer.trainable = True

    # Instanciation du modèle
    model = Sequential()

    model.add(pre_trained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units = 128, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = num_classes, activation = 'softmax')) 

    if summary:
        print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc']) 
    return model


early_stopping = EarlyStopping(
    patience=5,
    # Attendre 5 epochs avant application
    min_delta=0.01,
    # si au bout de 5 epochs la fonction de perte ne varie pas de 1%,
    # que ce soit à la hausse ou à la baisse, on arrête
    verbose=1,
    # Afficher à quel epoch on s'arrête
    mode='min',
    monitor='val_loss')

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    # si val_loss stagne sur 3 epochs consécutives selon la valeur min_delta
    min_delta=0.01,
    factor=0.1,  # On réduit le learning rate d'un facteur 0.1
    cooldown=4,  # On attend 4 epochs avant de réitérer
    verbose=1)

time_callback = TimingCallback()

# Générateur de données
train_datagen = ImageDataGenerator(
    shear_range=0.2,  # random application of shearing
    zoom_range=0.2,
    horizontal_flip=True)  # randomly flipping half of the images horizontally

test_datagen = ImageDataGenerator()


def define_fit_VGG16(X_train,
                   Y_train,
                   X_val,
                   Y_val,
                   n_retrained_vgg_layers = 4,
                   batch_size=64,
                   epochs=40,
                   summary=True):
    # Création, entraînement du modèle.
    
    train_dataset = train_datagen.flow(X_train, Y_train,
                                       batch_size=batch_size)
    val_dataset = test_datagen.flow(X_val, Y_val, batch_size=batch_size)


    # Construction d'un VGG16
    model = define_VGG16model(*X_train.shape[1:3], num_classes=Y_train.shape[1], n_retrained_vgg_layers = n_retrained_vgg_layers, summary=summary)

    # Entraînement du modèle
    history = model.fit(
            train_dataset,  # use augmented images for train
            steps_per_epoch=X_train.shape[0] // batch_size,
            validation_data=val_dataset,  # use augmented images for test
            epochs=epochs,
            callbacks=[
                        reduce_learning_rate,
                        early_stopping,
                        time_callback
                        ],
            verbose=True)
    return model, history
