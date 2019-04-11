from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.core import Dropout
from keras.layers import Convolution2D
##from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.np_utils import to_categorical

import serialize_test as srl
import image_process_test as ip
import os

def run_keras_lenet_model(train):
    model = Sequential()

    model.add(Convolution2D(8, 3, 3, activation='relu', input_shape=(ip.END_IMAGE_SIZE[0],
                                                                     ip.END_IMAGE_SIZE[1], 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu'))
##    model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(ip.END_IMAGE_SIZE[0],
##                                                                     ip.END_IMAGE_SIZE[1], 1)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(8, 3, 3, activation='relu'))
##    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(Convolution2D(8, 3, 3, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
##    model.add(Dense(84))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(len(train['train'][0]))
    print(len(train['train'][1]))
    print(len(train['valid'][0]))
    print(len(train['valid'][1]))
    
    model.fit(train['train'][0], train['train'][1],
              validation_data=(train['valid'][0], train['valid'][1]),
              shuffle=True, nb_epoch=50, batch_size = 124)
##    model.fit(train['train'][0], train['train'][1],
##              validation_data=(train['valid'][0], train['valid'][1]),
##              shuffle=True, epochs=10)

##    score = model.evaluate(train["test"][0], train['test'][1], verbose=0)
    score = model.evaluate(train["test"][0], train['test'][1], verbose=1, batch_size = 40)
    print(score)
    model.save('E:/autopipe_test/data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + '.h5')


if __name__ == "__main__":
##    data_pkl = os.getcwd() + '/data/serialized/autopipe-' + str(ip.END_IMAGE_SIZE[0]) + '-data.pkl'
    data_pkl = 'E:/autopipe_test/data/serialized/autopipe-' + str(ip.END_IMAGE_SIZE[0]) + '-data.pkl'
    data = srl.load_data(data_pkl)

    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data['test']

    normalized_data = {"train": [ip.preprocess(X_train), to_categorical(y_train)],
                       "valid": [ip.preprocess(X_valid), to_categorical(y_valid)],
                       "test": [ip.preprocess(X_test), to_categorical(y_test)]}

    run_keras_lenet_model(normalized_data)
