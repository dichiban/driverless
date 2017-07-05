import numpy as np
import cv2
import pandas as pd

import os


IMAGE_LOCATION = "data/AFTER_RACE/sorted/"
DATA_LOCATION = "data/AFTER_RACE/frame_data.csv"
IMAGE_SIZE = (128, 128)
DEPTH = 3

def load_images_from_folder(folder):
    """
    Loads images from folder
    :param folder: folder with images
    :return:
    images - List of images rgb
    """
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, str(i)+'.jpg'))
        if img is not None:
            resized = augment_data(img)
            images.append(resized)

    return np.asarray(images)

def norm_feature(x):
    x = x/255

    return x


def norm_label(y, left_min, left_max, right_min, right_max):
    """
    normalise the value
    :param y: original value
    :param left_min: original min
    :param left_max: original max
    :param right_min: desired min
    :param right_max: desired max
    :return:
    normalised steering angle
    """
    left_span = left_max - left_min
    right_span = right_max - right_min

    value_scaled = (y - left_min) / left_span

    return right_min + (value_scaled * right_span)


def get_canny(gray_frame):
    return cv2.Canny(gray_frame, 50, 200, apertureSize=3)


def get_yellow(img):
    """
    Highlight the yellow in the image
    :param img: original iamge
    :return:
    image with yellow highlighted
    """
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
    """
    :param img: original image
    :return:
    resized(128,128)/augmented image
    """
    y_start = int(len(img)*(0))
    y_end = int(len(img)*(1))
    x_start = int(len(img)*(0))
    x_end = int(len(img[0])*(1))
    #img = hsv_aug(img)
    #img = get_canny(img)
    aug = cv2.resize(img[y_start:y_end, x_start:x_end], IMAGE_SIZE)

    return aug


def load_data():
    """
    Loading the data
    :return
    X - training input data
    y - corresponding training output data
    """
    data_df = pd.read_csv(DATA_LOCATION)
    print(data_df['steering'].values)
    y = data_df['steering'].replace(['rawRxMotor'], '1500').apply(int).values
    #print(y)

    X = load_images_from_folder(IMAGE_LOCATION)
    print(X.shape)
    X = X.reshape(len(X), IMAGE_SIZE[0], IMAGE_SIZE[1], DEPTH)
    #y = data_df['steering'].values

    y_min = min(y)
    y_max = max(y)
    print(y_min)
    print(y_max)
    y = norm_label(y, y_min, y_max, -1, 1)

    return X, y

def save_aug_images():
    x, y = load_data()
    if not os.path.exists('aug_images'):
        os.makedirs('aug_images')
    print("done, processing, now saving")
    for i, img in enumerate(x):
        cv2.imwrite('aug_images/' + str(i) + '.jpg', img)

def hsv_aug(img):
    # load the image (1 means colour)
    frame = img

    # The mask will run through each blue colour here
    blues = [((95, 0, 220), (125, 50, 255)),
             ((100, 140, 165), (110, 255, 205)),
             ((95, 150, 150), (105, 255, 230)),
             ((95, 200, 150), (110, 255, 190)),
             ((100, 80, 95), (112, 220, 165)),
             ((95, 150, 160), (110, 240, 250)),
             ((105, 85, 85), (110, 240, 130))]

    # The mask will run through each yellow colour here
    yellows = [((20, 80, 200), (30, 130, 225)),
               ((25, 60, 150), (35, 105, 200)),
               ((25, 25, 180), (40, 70, 255)),
               ((20, 55, 180), (30, 90, 245)),
               ((10, 70, 130), (30, 120, 175)),
               ((25, 35, 80), (60, 95, 150)),
               ((30, 25, 165), (50, 60, 180)),
               ((40, 10, 235), (95, 25, 250)),
               ((95, 40, 170), (110, 130, 255)),
               ((15, 10, 240), (32, 50, 255))]

    # frame = adjust_gamma(frame, 1) # Ignore lel

    # Convert the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply first mask
    shapeMask = cv2.inRange(hsv, (35, 0, 210), (60, 35, 230))

    # Apply all of the yellow masks
    for yellow in yellows:
        shapeMask += cv2.inRange(hsv, yellow[0], yellow[1])

    # Apply all of the blue masks
    for blue in blues:
        shapeMask += cv2.inRange(hsv, blue[0], blue[1])

    # Erode/dilate the frame - helps to "push out" all the noise
    shapeMask = cv2.erode(shapeMask, None, iterations=1)
    shapeMask = cv2.dilate(shapeMask, None, iterations=3)

    frame_color = cv2.bitwise_and(hsv, hsv, mask=shapeMask)
    frame_color = cv2.addWeighted(frame_color, 1, frame_color, 1, 0)

    # Show us the masked image and move it
    # cv2.imshow(img[47:] + " Mask (color)", frame_color)
    # cv2.moveWindow(img[47:] + " Mask (color)", 50, 100)

    return frame_color

#save_aug_images()
# def get_latest_image(folder):
#     """
#     Obtain latest image from folder
#     :return
#     latest_resized - resized image of the latest image in a folder
#     """
#     latest = cv2.imread(os.path.join(folder, max((os.listdir(folder)))))
#     latest_resized = augment_data(latest)
#     cv2.imshow('sa', latest_resized)
#     cv2.waitKey()
#     latest_resized = latest_resized.reshape((1, 128, 128, 3))
#
#     return latest_resized

