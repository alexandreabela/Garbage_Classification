from params import *

## Ce qui est nécéssaire pour l'architecture du réseau de neurones
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def create_model(): 
    ## Taille de l'image en entrée
    input_shape = (TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[1],NUMBER_OF_CHANNELS)
    ## Architecture du réseau de neurones 
    model = Sequential()
    ## Conv1
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape,padding='same'))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## Conv2
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## Conv3
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## Conv4
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ## Conv5
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    ## Dense
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    ## Visualisation du modèle 
    model.summary()
    return model