import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import walk
from os.path import join
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from sklearn.utils import shuffle
X = []
Y = []

for root, dirs, files in walk("Face Database/"):
    for f in files:
        fullpath = join(root, f)

        Y.append(fullpath[15:17])

        imageA = mpimg.imread(fullpath)

        X.append(imageA)

X, Y = shuffle(X, Y)

Train_Data  = np.array(X[:600]) / 255 * 2 - 1
Train_Label = np.array(Y[:600])

Test_Data   = np.array(X[600:]) / 255 * 2 - 1
Test_Label  = np.array(Y[600:])

Train_Label = to_categorical(Train_Label, num_classes=51)
Test_Label = to_categorical(Test_Label, num_classes=51)

print("Training Data Shape: ", Train_Data.shape)
print("Training Label Shape: ", Train_Data.shape)

model = Sequential()
model.add(Conv2D(16, 3, activation="relu", input_shape=(240, 180, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(51, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

train_history=model.fit(Train_Data, Train_Label,
          batch_size=15,
          epochs=11,
          verbose=1,
          shuffle=True,
          validation_data=(Test_Data, Test_Label))
score = model.evaluate(Test_Data, Test_Label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  
show_train_history(train_history, 'acc', 'val_acc')  
show_train_history(train_history, 'loss', 'val_loss')  