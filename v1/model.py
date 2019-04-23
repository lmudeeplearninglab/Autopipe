#python script used to construct CNN learning model

#keras imports
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.core import Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.np_utils import to_categorical

from autopipe import serialize as srl
from autopipe import image_process as ip

#defining function which defines and runs learning model when passing a training set and K-fold parameter
def run_keras_lenet_model(train, K):
    #Model defined to have sequentially defined layers
    model = Sequential()

    #First layer: First convolution layer
    model.add(Convolution2D(filters = 6, kernel_size(5, 5), activation='relu', input_shape=(ip.END_IMAGE_SIZE[0],
                                                                     ip.END_IMAGE_SIZE[1], 1)))
    #Second layer: Max pooling layer
    model.add(MaxPooling2D())
    
    #Third Layer: Second convolution layer
    model.add(Convolution2D(filters = 6, kernel_size(5, 5), activation='relu'))
    
    #Fourth layer: Second Max pooling layer
    model.add(MaxPooling2D())

    #Flattens output of max pooling layer for input into Fully Connected Neural Network
    model.add(Flatten())
    
    #Fifth layer : First dropout layer
    model.add(Dropout(0.5))
    
    #Sixth Layer: First FC layer 120 nodes
    model.add(Dense(120, activation='relu'))
    
    #Seventh layer: Second dropout layer
    model.add(Dropout(0.5))
    
    #Eighth layer: Second FC layer 84 nodes
    model.add(Dense(84))
    
    #Ninth layer: Output layer
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Changed nb_epoch to epochs
    model.fit(train['train'][0], train['train'][1],
              validation_data=(train['valid'][0], train['valid'][1]),
              shuffle=True, epochs=10)

    score = model.evaluate(train["test"][0], train['test'][1], verbose=0)
    print(score)   
    
    #modified model save name to include train name
    model.save('../data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + train + '.h5')

#if model.py is executed perform this
if __name__ == "__main__":
    data_pkl = '../data/serialized/autopipe-' + str(ip.END_IMAGE_SIZE[0]) + '-data.pkl'
    data = srl.load_data(data_pkl)

    #defines dataset tuples
    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data['test']

    
    normalized_data = {"train": [ip.preprocess(X_train), to_categorical(y_train)],
                       "valid": [ip.preprocess(X_valid), to_categorical(y_valid)],
                       "test": [ip.preprocess(X_test), to_categorical(y_test)]}

    run_keras_lenet_model(normalized_data)
