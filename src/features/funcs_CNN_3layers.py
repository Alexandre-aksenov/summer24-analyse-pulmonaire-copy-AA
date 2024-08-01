"""
Functions for defining a CNN with 3 convolutional layers.
This module is used in:
train_CNN_128x128x2_3layers_save_v1.py

"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from timeit import default_timer as timer
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


def define_CNN_3layers_depth2(IMG_WIDTH,
                              IMG_HEIGHT,
                              num_classes,
                              summary=True):
    """
    Defines and compiles CNN a model.
    """
    # Instanciation du modèle
    model = Sequential()

    model.add(Conv2D(filters=30,
                     kernel_size=(5, 5),
                     padding='valid',
                     activation='relu',
                     input_shape=(IMG_WIDTH, IMG_HEIGHT, 2),
                     name='conv1'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

    # add a 2nd conv layer
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     padding='valid',
                     activation='relu',
                     name='conv2'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

    # add a 3rd conv layer
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     padding='valid',
                     activation='relu',
                     name='conv3'))

    model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    # add an intermediate dense layer
    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    if summary:
        print(model.summary())

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=['acc'])
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


def define_fit_CNN(X_train,
                   Y_train,
                   X_val,
                   Y_val,
                   batch_size=128,
                   epochs=20,
                   summary=True):
    # Création, entraînement du modèle.

    train_dataset = train_datagen.flow(X_train, Y_train,
                                       batch_size=batch_size)
    val_dataset = test_datagen.flow(X_val, Y_val, batch_size=batch_size)


# Construction d'un CNN Classique
#    model = define_CNN_3layers_depth2(IMG_WIDTH, IMG_HEIGHT, Y_train.shape[1], summary=summary)
    model = define_CNN_3layers_depth2(*X_train.shape[1:3], Y_train.shape[1], summary=summary)

# print(model.summary())

# Entraînement du modèle, this requires val_dataset

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
