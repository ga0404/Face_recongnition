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

model = load_model('train_model.h5')
model.summary()


for root, dirs, files in walk("test_data/"):
    for f in files:
        fullpath = join(root, f)

        Y = fullpath[-9:-7]

        X = mpimg.imread(fullpath) / 255 * 2 - 1


        #fullpath = str(sys.argv[1])
        # print("\nPlease input data name:")
        # path=input()
        # fullpath=("./Face Database/"+path)
        # print(fullpath)

        Test_Label  = np.array(Y)

        Test_Data = X[np.newaxis, ...]


        print("Training Data Shape: ", X.shape)

        
        result = model.predict(Test_Data)

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
        if(sorted_result_dict[0][0]<10):
            print("0"+str(sorted_result_dict[0][0]))
        else:
            print(sorted_result_dict[0][0])

        print('\nTop 5 candidate: ')
        for tp in sorted_result_dict:
            cnt += 1
            print(str(tp[0])+' Possibility: %0.6f' %(tp[1]))
            if cnt >= 5:
                break
        wait = input("Press Any Key")
