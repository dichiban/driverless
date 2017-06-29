import numpy as np
import cv2
import pandas as pd

import os


IMAGE_LOCATION = "2017-06-29-1"
DATA_LOCATION = "frame_data1.csv"


NUM_IMAGES = 4491
def load_images_from_folder(folder):
    """
    Loads images from folder
    :return
    images - List of images rgb
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resized = augment_data(img)
            images.append(resized)

    return np.asarray(images)

def norm_label(y, left_min, left_max, right_min, right_max):
    left_span = left_max - left_min
    right_span = right_max - right_min

    value_scaled = (y - left_min) / left_span

    return right_min + (value_scaled * right_span)

def get_canny(gray_frame):
    return cv2.anny(gray_frame, 50, 200, apertureSize=3)

def get_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define Ranges for yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    # Create a logical mask to get binary values based on pre-determined ranges
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)  # sets values in range to 1 and others to 0
    # Calculate resultant image of mask over input
    yellow_res = cv2.bitwise_and(img, img, mask=yellow_mask)  # reveals values in all pixels in both mask and frame
    y_res = cv2.addWeighted(yellow_res, 1, yellow_res, 1, 0)
    return y_res

def augment_data(img):
    y_start = int(len(img)*(1/3))
    y_end = int(len(img)*(2/3))
    x_start = int(len(img)*(0))
    x_end = int(len(img[0])*(1))
    img = get_yellow(img)
    aug = cv2.resize(img[y_start:y_end, x_start:x_end], (128, 128))
    return aug

def load_data():
    """
    Loading the data
    :return
    X - training input data
    y - corresponding training output data
    """
    data_df = pd.read_csv(DATA_LOCATION)

    X = load_images_from_folder(IMAGE_LOCATION)
    X = X.reshape(len(X), 128, 128, 3)
    y = data_df['throttle'].values
    y_min = min(y)
    y_max = max(y)
    y = norm_label(y, y_min, y_max, -1, 1)
    return X, y

def get_latest_image(folder):
    """
    Obtain latest image from folder
    :return
    latest_resized - resized image of the latest image in a folder
    """
    latest = cv2.imread(os.path.join(folder, max((os.listdir(folder)))))
    latest_resized = augment_data(latest)
    cv2.imshow('sa', latest_resized)
    cv2.waitKey()
    latest_resized = latest_resized.reshape((1, 128, 128, 3))

    return latest_resized

