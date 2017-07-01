from keras.models import model_from_yaml
import serial
import data
import cv2
import threading
from time import time

ser = serial.Serial('/dev/ttyUSB0', 115200)

yaml_file = open('IT_WORKS/model0.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

model = model_from_yaml(loaded_model_yaml)
model.load_weights('IT_WORKS/model_weights0.h5')

cam = cv2.VideoCapture(0)
frame = 0

start = time()
i = 0

while 1:
    s, im = cam.read()
    im = data.augment_data(im)
    im = im.reshape((1, 128, 128, 3))
    y = model.predict(im)
    frame += 1
    y = data.norm_label(y, -1, 1, 1044, 1750)
    fin = time()
    if fin - start >= 1:
        i+=1
        fps = frame/i
        print(fps)
        start = fin
    ser.write('steer,'+str(y)+"\n")
