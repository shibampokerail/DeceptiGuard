import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python import keras
from keras.models import Model as keras_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import Adam
from keras.models import Model
import os
import random

from win11toast import toast


# Channels = amount of color channels (red, green, blue) 3 because we are using coloured images
image_dimensions = {'height':256, 'width':256, 'channels':3}

IMG_SIZE = 256

class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


class Meso1(Classifier):
    """
    Feature extraction + Classification
    """

    def __init__(self, learning_rate=0.001, dl_rate=1):
        self.model = self.init_model(dl_rate)
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self, dl_rate):
        x = Input(shape=(IMG_SIZE,IMG_SIZE, 3))

        x1 = Conv2D(16, (3, 3), dilation_rate=dl_rate, strides=1, padding='same', activation='relu')(x)
        x1 = Conv2D(4, (1, 1), padding='same', activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        y = Flatten()(x1)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        return keras_model(inputs=x, outputs=y)


# Create a MesoNet class using the Classifier

class Meso4(Classifier):

    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def init_model(self):
        #need to pass 3 dimensions of image data
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

def load_and_predict(img_path):
    meso = Meso4()
    meso.load("./weights/Meso4_DF.h5")
    dataGenerator = ImageDataGenerator(rescale=1./255)

    generator = dataGenerator.flow_from_directory(
        './data/',
        target_size=(256,256),
        batch_size=1,
        class_mode='binary')#jjust two classes real or deepfake

    directory = 'test'

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out non-image files (if any)
    image_files = [f for f in all_files if f.endswith('.jpg') or f.endswith('.png')]

    # Select a random image file
    # random_image_file = random.choice(image_files)



        # random_image_path = os.path.join(directory, random_image_file)


    # image_path = os.path.join(directory, random_image_file)
    # custom_image_path = random_image_path
    custom_image_path = img_path
    custom_image = load_img(custom_image_path, target_size=(256, 256))
    custom_image_array = img_to_array(custom_image)
    custom_image_array = custom_image_array.reshape((1,) + custom_image_array.shape)
    custom_image_array /= 255.0

    print(f"Predicted Likelihood: {meso.predict(custom_image_array)[0][0]:.4f}")
    prediction = ""
    if (meso.predict(custom_image_array)[0][0]<0.8):
        print("Deepfake")
        prediction = "Deepfake"

        toast("DeepFakeDetector", "There is a deepfake image on your screen!",  on_click=lambda args:os.startfile(img_path) )
    else:
        print("Real")
        prediction = "Real"


    print(f"Actual Label: {custom_image_path}")
    # print(f"\nCorrect prediction: {round(meso.predict(X)[0][0])==y[0][0]}")

    return prediction



def show_img(path):
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.show()