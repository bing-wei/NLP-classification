#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from keras.preprocessing import text
from keras import utils
import matplotlib.pyplot as plt

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
sample_upload = pd.read_csv('Data/sample_upload.csv')
#%% 
x, y = train['tweet'], train['class']
num_classes = 3
max_words = 10000

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(x) 

X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y)

y_test_label = y_test
y_train_label = y_train

x_train = tokenize.texts_to_matrix(X_train)
x_test = tokenize.texts_to_matrix(X_test)

y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)



#%% 模型設定

batch_size = 32
epochs = 100
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(x_test, y_test))


#%% 查看分數
prediction = model.predict_classes(x_test)
y_prediction = model.predict_classes(x_train)
y_pred_bool = utils.to_categorical(prediction, num_classes)


print('-------------------------train-------------------------')
print(pd.crosstab(y_train_label, y_prediction, rownames=['label'], colnames=['predict']))
print('-------------------------test--------------------------')
print(pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))
print('-------------------------score--------------------------')
print(classification_report(y_test, y_pred_bool))
#%%
loss	= history.history.get('loss')
acc 	= history.history.get('val_accuracy')
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='loss')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='acc')
plt.title('Accuracy')
plt.show()

