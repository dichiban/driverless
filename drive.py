from keras.models import model_from_yaml
import data

x_train, y_train = data.load_data()
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

model = model_from_yaml(loaded_model_yaml)
model.load_weights("model_weights.h5")

while 1:
    pic = data.get_latest_image("Ch2_001/center")

    y = model.predict(pic)

    print(y)