import  numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras import optimizers
from keras import backend as K

import data

def customized_loss(y_true, y_pred, loss='elucidean'):
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
                    + K.sum(K.square(y_pred), axis=-1)/2 * L2_norm_cost

    elif loss == 'elucidean':
        val = K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))

    return  val

def create_model(keep_prob = 0.8):
    model = Sequential()

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(100, 100, 3)))
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
    model.add(Dense(1, activation='softsign'))

    return model

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

if __name__ == '__main__':
    # Load Training Data
    x_train, x_valid, y_train, y_valid = data.load_data()

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model()
    model.compile(loss='mean_squared_error', optimizer=optimizers.adam(), metrics=[soft_acc])
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs, shuffle=True)

    loss, acc = model.evaluate(x_valid, y_valid, verbose=0)
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    model.save_weights('model_weights.h5')