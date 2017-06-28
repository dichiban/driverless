import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

import os


IMAGE_LOCATION = "Ch2_001/center"
DATA_LOCATION = "Ch2_001/final_example.csv"


NUM_IMAGES = 4491
def load_images_from_folder(folder):
    """
    Loads images from folder
    :return
    images - List of images in numbers
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            #resized = cv2.resize(img, (128, 128))
            resized = cv2.resize(img[240:480, 0:640], (128, 128))
            #resized = img[240:480, 0:640]
            images.append(resized)

    return np.asarray(images)


def load_data():
    """
    Loading the data
    :return
    X_train - training input data
    X_valid - validate input data
    y_train - corresponding training output data
    y_valid - corresponding validate output data
    """
    data_df = pd.read_csv(DATA_LOCATION)

    X = load_images_from_folder(IMAGE_LOCATION)
    y = data_df['steering_angle'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    y_train = np.reshape(y_train, (y_train.size, 1))
    y_valid = np.reshape(y_valid, (y_valid.size, 1))

    return X, y
    #return X_train, X_valid, y_train, y_valid

def get_latest_image(folder):
    latest = cv2.imread(os.path.join(folder, max((os.listdir(folder)))))
    latest_resized = cv2.resize(latest[240:480, 0:640], (128, 128))
    latest_resized = latest_resized.reshape((1, 128, 128, 3))

    return latest_resized