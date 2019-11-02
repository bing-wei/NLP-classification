# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:34:47 2019

@author: wei
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_upload = pd.read_csv('sample_upload.csv')
#%%

train['tweet'] = [entry.lower() for entry in train['tweet']]
train['tweet']= [word_tokenize(entry) for entry in train['tweet']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(train['tweet']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    train.loc[index,'text_final'] = str(Final_words)


test['tweet'] = [entry.lower() for entry in test['tweet']]
test['tweet']= [word_tokenize(entry) for entry in test['tweet']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(test['tweet']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    test.loc[index,'text_final'] = str(Final_words)

#%%
from sklearn.utils import class_weight
x, y = train['text_final'], train['class']
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y)



max_words = 3000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(x) 

x_train = tokenize.texts_to_matrix(X_train)
x_test = tokenize.texts_to_matrix(X_test)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
num_classes = np.max(y_train) +1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


batch_size = 32
epochs = 2


model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              


history = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(x_test, y_test)
                       ,class_weight=class_weights
                       )
          


from sklearn.metrics import classification_report

y_pred = model.predict_classes(x_test)
y_pred_bool = utils.to_categorical(y_pred, num_classes)

print(classification_report(y_test, y_pred_bool))
