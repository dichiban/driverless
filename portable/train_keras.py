import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import optimizers
from keras import backend as K

import data


INPUT = (128, 128, 3)

def create_model(keep_prob = 0.8):
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


def elucidean_loss(y_true, y_pred):
    loss = K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))
    return loss

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

if __name__ == '__main__':
    # Load Training Data
    x_train, y_train= data.load_data()

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model()


    sgd = optimizers.SGD(lr=0.0001)
    adam = optimizers.Adam(lr=0.0001)

    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)

    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    model.save_weights('model_weights.h5')


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

