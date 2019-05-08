#python script used to construct CNN learning model

#keras imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.core import Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np

#commented out to run model.py locally from command line
#from autopipe import serialize as srl
#from autopipe import image_process as ip

#modified for local run
import serialize as srl
import image_process as ip

from sklearn.model_selection import KFold

#defining function which defines and runs learning model when passing a training set and K-fold parameter
def run_keras_lenet_model(train, iteration, epos):
    #Model defined to have sequentially defined layers
    model = Sequential()

    #First layer: First convolution layer
    model.add(Convolution2D(filters = 6, kernel_size=(5, 5), activation='relu', input_shape=(ip.END_IMAGE_SIZE[0],
                                                                     ip.END_IMAGE_SIZE[1], 1)))
    #Second layer: Max pooling layer
    model.add(MaxPooling2D())
    
    #Third Layer: Second convolution layer
    model.add(Convolution2D(filters = 6, kernel_size=(5, 5), activation='relu'))
    
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
              validation_data=(train['val'][0], train['val'][1]),
              shuffle=True, epochs=epos, verbose=1)

    score = model.evaluate(train["test"][0], train['test'][1], verbose=1)
    print("Test loss, Test accuracy: ", score)   
    
    #modified model save name to include train name
    #commented out for local run
	#model.save('../data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + train + '.h5')
    model.save('data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + 'train' + str(iteration) + 'epos' + str(epos) + '.h5')

#if model.py is executed perform this
if __name__ == "__main__":
    #commented out for local run
	#data_pkl = '../data/serialized/autopipe-' + str(ip.END_IMAGE_SIZE[0]) + '-data.pkl'
    data_pkl = 'data/serialized/autopipe-' + str(ip.END_IMAGE_SIZE[0]) + '-data.pkl'
    data = srl.load_data(data_pkl)
 
    #defines dataset tuples
    X1, y1 = data["train"]
    X2, y2 = data['test']
    
	#define arrays
    X_train = np.array(X1)
    y_train = np.array(y1)
    X_test = np.array(X2)
    y_test = np.array(y2)
	
    K = 10
    counter = 0
    #implement k-fold cross validation
    kf = KFold(n_splits=K, shuffle=True)
    for train_index, test_index in kf.split(X_train):
         counter = counter + 1
		 #debugging
         #print("Train_index:", train_index, "    Test index:", test_index)
         X_t, X_v = X_train[train_index], X_train[test_index]
         y_t, y_v = y_train[train_index], y_train[test_index]  
    
         normalized_data = {"train": [ip.preprocess(X_t), to_categorical(y_t)],
                            "val": [ip.preprocess(X_v), to_categorical(y_v)],
                            "test": [ip.preprocess(X_test), to_categorical(y_test)]}
         
         #run_keras_lenet_model(normalized_data, counter, 10)
         run_keras_lenet_model(normalized_data, counter, 25)
         #run_keras_lenet_model(normalized_data, counter, 30)
         #run_keras_lenet_model(normalized_data, counter, 40)
         #run_keras_lenet_model(normalized_data, counter, 50)
         #run_keras_lenet_model(normalized_data, counter, 60)
         #run_keras_lenet_model(normalized_data, counter, 70)
         #run_keras_lenet_model(normalized_data, counter, 80)
         #run_keras_lenet_model(normalized_data, counter, 90)
		 #run_keras_lenet_model(normalized_data, counter, 100)
