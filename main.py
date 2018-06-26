import tensorflow
import keras
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping

import multiprocessing as mp
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.image as mpimg
from random import shuffle
from skimage.transform import resize

# data process and label
picarray = np.empty((650, 240, 180, 3))
labelarray = np.zeros((650, 50))

count = 0
for person in range(50):
    for pic in range(15):
        if pic ==4 or pic == 8:
               
            continue
            pic = mpimg.imread('Face Database/s' + str(person + 1).zfill(2) +'_' + str(pic + 1).zfill(2) + '.jpg')
            pic = resize(pic, (240, 180))
            picarray[count] = pic
            count = count + 1
            labelarray[count][person] = 1
     



print(picarray.shape)
print(labelarray.shape)

# data process and label
