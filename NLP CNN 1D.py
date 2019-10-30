# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:06:03 2019

@author: wei
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.preprocessing import text
import numpy as np
from keras import utils
from keras.layers import Dense, Activation, Dropout

train = pd.read_csv('C://Users//wei//Desktop//python//twitter//train.csv')
test = pd.read_csv('C://Users//wei//Desktop//python//twitter//test.csv')
sample_upload = pd.read_csv('C://Users//wei//Desktop//python//twitter//sample_upload.csv')
x, y = train['tweet'], train['class']
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y)

#%%
max_words = 2000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(X_train) 

x_train = tokenize.texts_to_matrix(X_train)
x_test = tokenize.texts_to_matrix(X_test)

num_classes = np.max(y_train) +1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv1D(64, 3, border_mode='same', input_shape=(10, 32)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(X_test, y_test, batch_size=16)