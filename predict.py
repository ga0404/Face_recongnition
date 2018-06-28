# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 07:56:18 2018

@author: louis
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from os.path import join
import numpy as np
import operator
import sys

#print(print("\n".join(sys.argv)))

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.utils import shuffle

X = []
Y = []

#fullpath = str(sys.argv[1])
print("\nPlease input data name:")
path=input()
fullpath=("./Face Database/"+path)
print(fullpath)

X = mpimg.imread(fullpath) / 255 * 2 - 1
X = X[np.newaxis, ...]
Y = fullpath[-9:-7]


print("Training Data Shape: ", X.shape)

model = load_model('train_model.h5')
model.summary()
result = model.predict(X)
print("Predict result: ")
print(result[0])
print("\nTrue Answer: ")
print(Y)

result_dict = {}
for i in range(len(result[0])):
    result_dict[i] = float(result[0][i])
sorted_result_dict = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)
cnt = 0
print("\nPredict Answer:")
print(sorted_result_dict[0][0])
print('\nTop 5 candidate: ')
for tp in sorted_result_dict:
    cnt += 1
    print(str(tp[0])+' Possibility: %0.6f' %(tp[1]))
    if cnt >= 5:
        break
