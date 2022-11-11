

import tensorflow as tf # used to access argmax function
from tensorflow import keras # for building Neural Networks
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator # For data augmentation
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop
from tensorflow.keras import regularizers

import image_augmentation as aug

def eyespot_detection_model():
    model = Sequential([
        Conv2D(32,(7,7),activation='relu', padding="same",input_shape=(30,30,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D((2,2)),
    #    Dropout(0.2),
        Flatten(),
        Dense(120, activation="relu",
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)),
        
        Dense(60, activation="relu"),
        Dropout(0.45),
        Dense(units=2, activation='softmax')
    ])
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=['accuracy'])

    return model



def pixel_detection_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(5, 5, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax')) 

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    return model