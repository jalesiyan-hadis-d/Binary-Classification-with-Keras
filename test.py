# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:04:12 2019

@author: h.jalisian
"""
import pandas
from sklearn.preprocessing import scale
from keras.models import model_from_json

Labels=["B", "M"]
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


dataframe = pandas.read_csv("test.csv", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
Xtest = dataset[:,1:31].astype(float)
SXtest=scale(Xtest)
loaded_model = load_model()
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'MSE'])
ynew=(loaded_model.predict_classes(SXtest)).astype(int)
print(ynew)

