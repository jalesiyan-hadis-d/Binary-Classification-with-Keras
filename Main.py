# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:25:40 2019

@author: h.jalisian
"""

# Train model and make predictions
import numpy
import pandas
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale


# fix random seed for reproducibility
seed = 7

# load dataset
dataframe = pandas.read_csv("data1.csv", header=0)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,2:32].astype(float)
Y = dataset[:,1]



# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def Create_Model():
    model = Sequential()
    model.add(Dense(output_dim=16, input_dim=30, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(p=0.1))
    model.add(Dense(output_dim=16,kernel_initializer='uniform',activation='relu'))
    model.add(Dropout(p=0.1))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'MSE'])
    return model
def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    
#Later
def load_model():
   
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

#Code2:onenote
# evaluate baseline model with standardized dataset
numpy.random.seed(seed)


#ُStandard
SX=scale(X)

X_train, X_test, Y_train, Y_test = train_test_split(SX, encoded_Y, test_size=0.2, random_state=seed)

model=Create_Model()

# Fit the model
model.fit(X_train, Y_train, epochs=20, batch_size=5, verbose=0)
# evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%,\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100, model.metrics_names[2], scores[2]*100))
#print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
ynew=(model.predict_classes(X_test)).astype(int)
# save
save_model(model)
"""
# load
loaded_model = load_model()

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'MSE'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%,\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100, loaded_model.metrics_names[2], scores[2]*100))
"""
