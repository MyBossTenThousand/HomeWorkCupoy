# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:11:49 2020

@author: user
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model


##kernel size=(6,6)
##kernel數量：32

## Same padding、strides=(1,1)
classifier=Sequential()

inputs = Input(shape=(13,13,1))
x=Convolution2D(32,6,strides=(1,1),padding="Same")(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Same padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,6,strides=(2,2),padding="Same")(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Valid padding、strides=(1,1)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,6,strides=(1,1),padding="Valid")(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
## Valid padding、strides=(2,2)
classifier=Sequential()
inputs = Input(shape=(13,13,1))
x=Convolution2D(32,6,strides=(2,2),padding="Valid")(inputs)
model = Model(inputs=inputs, outputs=x)
model.summary()
