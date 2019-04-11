from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.core import Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.np_utils import to_categorical

from autopipe import serialize as srl
from autopipe import image_process as ip


def run_keras_lenet_model(train):
    model = Sequential()

    model.add(Convolution2D(6, 5, 5, activation='relu', input_shape=(ip.END_IMAGE_SIZE[0],
                                                                     ip.END_IMAGE_SIZE[1], 1)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train['train'][0], train['train'][1],
              validation_data=(train['valid'][0], train['valid'][1]),
              shuffle=True, nb_epoch=10)

    score = model.evaluate(train["test"][0], train['test'][1], verbose=0)
    print(score)
    model.save('../data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + '.h5')


if __name__ == "__main__":
    data_pkl = '../data/serialized/autopipe-' + str(ip.END_IMAGE_SIZE[0]) + '-data.pkl'
    data = srl.load_data(data_pkl)

    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data['test']

    normalized_data = {"train": [ip.preprocess(X_train), to_categorical(y_train)],
                       "valid": [ip.preprocess(X_valid), to_categorical(y_valid)],
                       "test": [ip.preprocess(X_test), to_categorical(y_test)]}

    run_keras_lenet_model(normalized_data)