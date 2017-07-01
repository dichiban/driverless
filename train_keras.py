import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import optimizers
from keras import backend as K

from sklearn.model_selection import StratifiedKFold


import data


INPUT = (128, 128, 3)

def create_model(keep_prob = 0.7):
    model = Sequential()

    # Nvidia end to end model with dropouts
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=INPUT))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='tanh'))

    return model


def acc(y_true, y_pred):
    return 1 - K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))


if __name__ == '__main__':
    # Load Training Data
    x_train, y_train= data.load_data()

    print(x_train.shape[0], 'train samples')

    seed = 7
    np.random.seed(seed)

    # Training loop variables
    epochs = 100
    batch_size = 50

    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

    sgd = optimizers.SGD(lr=0.0001)
    adam = optimizers.Adam(lr=0.0001)

    cvscores =[]

    for i, (train, test) in enumerate(kfold.split(x_train, y_train)):
        model = create_model()
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=[acc])
        model.fit(x_train[train], y_train[train], batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

        scores = model.evaluate(x_train[test], y_train[test], verbose=0)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

        model_yaml = model.to_yaml()
        with open("model"+str(i)+".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)

        model.save_weights('model_weights'+str(i)+'.h5')

    for score in cvscores:
        print(score)

    # Plot the accuracy + loss
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['soft_acc'])
    # plt.plot(history.history['val_acc'])
    # plt.plot(history.history['val_soft_acc'])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model accuracy/loss')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['accuracy', 'soft_accuracy', 'val_accuracy', 'val_soft_accuracy', 'loss', 'val_loss'])
    # plt.savefig('model_accuracy_loss.png')


    #Save architecutre and weights

