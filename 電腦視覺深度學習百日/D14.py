# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:41:34 2020

@author: user
"""

from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import BatchNormalization
from keras.layers import Activation

input_shape = (32, 32, 3)

model = Sequential()

##  Conv2D-BN-Activation('sigmoid') 

#BatchNormalization主要參數：
#momentum: Momentum for the moving mean and the moving variance.
#epsilon: Small float added to variance to avoid dividing by zero.

model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
model.add(BatchNormalization()) 
model.add(Activation('sigmoid'))


##、 Conv2D-BN-Activation('relu')
model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
model.add(BatchNormalization()) 
model.add(Activation('relu'))


model.summary()