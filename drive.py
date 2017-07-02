from keras.models import model_from_yaml
import serial
import data
import cv2

from time import time

#ser = serial.Serial('/dev/ttyUSB0', 115200)

yaml_file = open('IT_WORKS/model_no_aug.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

model = model_from_yaml(loaded_model_yaml)
model.load_weights('IT_WORKS/model_weights_no_aug.h5')

cam = cv2.VideoCapture(0)
frame = 0

start = time()
i = 0
fps = 0

# This is the "loop" that is run each time a frame is capture/a prediction is made
def run_CV(warming):
    #You have to "pull in" the globally scoped variables
    global model, cam, frame, start, i, fps
    global frame
    s, im = cam.read()
    im = data.augment_data(im)
    im = im.reshape((1, 128, 128, 3))
    y = model.predict(im)
    frame += 1
    y = data.norm_label(y, -1, 1, 1044, 1750)
    fin = time()
    if fin - start >= 1:
        i += 1
        fps = round(frame / i)
        start = fin
    if warming:
        pass

    else:
        #ser.write(('steer,'+str(y)+"\n").encode())
        print("FPS: {} Steering: {}".format(fps, int(y)))

print("\"Pre Warming\" the TensorFlow model...")

for x in range(200):
    run_CV(1)


# Wait for us to press a key before beginning.  Note: Starts control immediately
key_wait = input("Press any key to start Autonomous control *RIGHT NOW*...")

#ser.write(b'motor,1545')

while 1:
    run_CV(0)