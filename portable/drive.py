from keras.models import model_from_yaml
import data
import matplotlib.pyplot as plt


# Load the model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()

model = model_from_yaml(loaded_model_yaml)
model.load_weights("model_weights.h5")

#Get the latest image from a folder and predict the steering angle

x, y = data.load_data()
y_min = min(y)
y_max = max(y)
y_pred = []

for img in x:
    img = img.reshape(1, 128, 128, 3)
    pred_angle = model.predict(img)
    y_pred.append(pred_angle[0][0])

print(len((y)))
print(len(y_pred))
plt.plot(y)
plt.plot(y_pred)
plt.title('Pred vs Real')
plt.ylabel('Steering angle')
plt.xlabel('frame')
plt.legend(['real', 'pred'])
plt.show()
