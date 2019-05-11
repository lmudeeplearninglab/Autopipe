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
import matplotlib.pyplot as plt

#commented out to run model.py locally from command line
#from autopipe import serialize as srl
#from autopipe import image_process as ip

#modified for local run
import serialize as srl
import image_process as ip

from sklearn.model_selection import KFold

#function trains model
def train_model(train, iteration, epos, weights=""):
    model = create_model()
    if weights:
        print("Loading weights from file")
        model.load_weights('./' + weights)
    else:
        print("Weights undefined")
        
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Changed nb_epoch to epochs
    history = model.fit(train['train'][0], train['train'][1],
              validation_data=(train['val'][0], train['val'][1]),
              shuffle=True, epochs=epos, verbose=2)
    
    loss_values = history.history['loss']
    val_loss_values= history.history['val_loss']
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    epochs = range(1, len(loss_values)+1)
    
    #plot training accuracy and loss
    plt.plot(epochs, loss_values, 'b--', epochs, val_loss_values, 'r-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(epochs, accuracy, 'b--', epochs, val_accuracy, 'r-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    #modified model save name to include train name
    #commented out for local run
	#model.save('../data/models/lenet_' + str(ip.END_IMAGE_SIZE[0]) + train + '.h5')
    model.save('data/models/lenet_' + str(ip.END_IMAGE_SIZE[1]) + 'train' + str(iteration) 
               + 'epos' + str(epos) + '.h5')
    weight_file = 'data/models/lenet_' + str(ip.END_IMAGE_SIZE[1]) + 'train' + str(iteration) + 'epos' + str(epos) + '.h5'
    return weight_file

#function test model			   
def test_model(train, weightfile):
    model = create_model()
    model.load_weights(weightfile)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(train["test"][0], train['test'][1], verbose=2)
    print("Test loss, Test accuracy: ", score)   
    return
	
#function defines model	
def create_model():
    #Model defined to have sequentially defined layers
    model = Sequential()

    #First layer: First convolution layer; input_shape(height, width, channels)
    model.add(Convolution2D(filters = 6, kernel_size=(3, 3), activation='relu', input_shape=(ip.END_IMAGE_SIZE[1],ip.END_IMAGE_SIZE[0], 1)))
    model.add(MaxPooling2D())					#Second layer: Max pooling layer
    model.add(Convolution2D(filters = 6, kernel_size=(3, 3), activation='relu'))	    #Third Layer: Second convolution layer
    model.add(MaxPooling2D())					#Fourth layer: Second Max pooling layer
    model.add(Flatten())						#Flattens output of max pooling layer for input into Fully Connected Neural Network
    model.add(Dropout(0.7))						#Fifth layer : First dropout layer
    model.add(Dense(120, activation='relu'))	#Sixth Layer: First FC layer 120 nodes
    #model.add(Dropout(0.5))						#Seventh layer: Second dropout layer
    model.add(Dense(84))    					#Eighth layer: Second FC layer 84 nodes
    model.add(Dense(2, activation='softmax'))	#Ninth layer: Output layer
    return model
	
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
	
    K = 5
    counter = 0
    weighttest = ""
    #implement k-fold cross validation
    kf = KFold(n_splits=K, shuffle=False)
    for train_index, test_index in kf.split(X_train):
         counter = counter + 1
		 #debugging
         #print("Train_index:", train_index, "    Test index:", test_index)
         X_t, X_v = X_train[train_index], X_train[test_index]
         y_t, y_v = y_train[train_index], y_train[test_index]  
    
         normalized_data = {"train": [ip.preprocess(X_t), to_categorical(y_t)],
                            "val": [ip.preprocess(X_v), to_categorical(y_v)],
                            "test": [ip.preprocess(X_test), to_categorical(y_test)]}
         weighttest = train_model(normalized_data, counter, 15, weighttest)
         test_model(normalized_data, weighttest)
        
