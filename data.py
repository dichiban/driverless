import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

import os


IMAGE_LOCATION = "Ch2_001\center"
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
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            resized = cv2.resize(img, (100, 100))
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
    return X_train, X_valid, y_train, y_valid

