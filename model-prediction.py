from PIL import Image # read in images
import numpy as np 
from keras.models import load_model
import os
import cv2

imgs = [] # list for images
cats = [] # list for categories

def images_to_array(folder, name):
    for image in os.listdir(folder):
        loaded_image = Image.open(os.path.join(folder, image))
        resized_image = Image.Image.resize(loaded_image, [100, 100])
        image_array = np.array(resized_image)
        blurred_image = cv2.blur(image_array, (2, 2))
        imgs.append(image_array)
        cats.append(name_encode[name])
        imgs.append(blurred_image)
        cats.append(name_encode[name])


def predict(index_number):
    mod = load_model('./sofa_training_CNN.h5')
    img = (np.array(imgs[index_number]) -127.5)/127.5
    img = img.reshape(1, 100, 100, 3)
    prediction = mod.predict_classes(img)
    print(prediction[0])

def predict_new(path):
    mod = load_model('./sofa_training_CNN.h5')
    loaded_image = Image.open(path)
    resized_image = Image.Image.resize(loaded_image, [100, 100])
    img = (np.array(resized_image) -127.5)/127.5
    img = img.reshape(1, 100, 100, 3)
    prediction = mod.predict_classes(img)
    print(prediction[0])


## set up lists
all_rh = './assets/sofas/all_rh_corner'
all_mid_cent = './assets/sofas/all_mid_cent'
name_encode = {'corner': 0, 'mid_cent': 0,}

images_to_array(all_rh, 'corner')
images_to_array(all_mid_cent, 'mid_cent')

# prediction on data we have, a good test to 
# take an image we have and test if its predicted correctly
#predict(30)
#predict(200)
#predict(len(imgs)-10)
#predict(len(imgs)-50)

# Load an new mage to test

# chesterfield = './assets/sofas/test/chesterfield.jpg'
# rounded = './assets/sofas/test/rounded.jpg'
# sofabed = './assets/sofas/test/sofabed.jpg'
difficult_rh = './assets/sofas/test/difficult_rh.jpg'
difficult_mid_cent = './assets/sofas/test/difficult_mid_cent.jpg'

# predict_new(chesterfield)
# predict_new(rounded)
# predict_new(sofabed)
predict_new(difficult_mid_cent)
predict_new(difficult_rh)
