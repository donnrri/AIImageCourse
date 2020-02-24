from PIL import Image
import numpy as np 
import  matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPool2D
from keras.models import Sequential, load_model
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import cv2


imgs = [] # images
cats = [] # images

rh_corner_sofa= './assets/sofas/right_hand_corner'
mid_cent = './assets/sofas/mid_centuary'
test = './assets/sofas/test'
name_encode = {'corner': 0, 'mid_cent': 0}

def images_to_array(folder, name):
    for image in os.listdir(folder):
        loaded_image = Image.open(os.path.join(folder, image))
        resized_image = Image.Image.resize(loaded_image, [100, 100])
        image_array = np.array(resized_image)
        imgs.append(image_array)
        cats.append(name_encode[name])
        # blur image to increase size of dataset
        blurred_image = cv2.blur(image_array (2, 2))
        imgs.append(blurred_image)
        cats.append(name_encode[name])


def show_image(index):
    plt.imshow(imgs[index])
    plt.show()


def convert_webp_jpg(path):
    return Image.open(path).convert("RGB)")


images_to_array(rh_corner_sofa, 'corner')
images_to_array(mid_cent, 'mid_cent')
#images_to_array(test, 'mid_cent')

categorical_cats = to_categorical(cats, num_classes = 2)
imgs = (np.array(imgs) - 127.5)/127.5


model = Sequential()
model.add(Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(100, (5,5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(100, (5,5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(124))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.summary()

mid_cent_train, mid_cent__test, corner_train, corner_test = train_test_split(imgs, categorical_cats, test_size=0.1)

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
h = model.fit(mid_cent_train, corner_train, batch_size=10, epochs=20, validation_data=(mid_cent__test, corner_test))
model.save('sofa_training_CNN-h5')

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('CNN accuracy train / test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')