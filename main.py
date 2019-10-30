# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:01:00 2019

@author: wei
"""
#%%
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


#%%
train = pd.read_csv('C://Users//wei//Desktop//python//twitter//train.csv')
test = pd.read_csv('C://Users//wei//Desktop//python//twitter//test.csv')
sample_upload = pd.read_csv('C://Users//wei//Desktop//python//twitter//sample_upload.csv')
op = open('C://Users//wei//Desktop//python//twitter//stop.txt', "r")
stop_word = op.read().splitlines()
op.close()
#%%  
def clean_en_text(ce):
    comp = re.compile("[^A-Z^a-z^0-9^ ^'^]")
    return comp.sub(' ', ce)
#%% 清除符號
x = train['tweet'].to_list()
for i in range(len(x)) :
    x[i] = clean_en_text(x[i])
train['st'] = x
#%% 全部轉小寫
for i in range(len(train['st'])):
    train['st'][i] = str.lower(train['st'][i])

#%% st單字拆解
clear = train['st']
for  i in range(len(clear)) :
    clear[i] = [t for t in word_tokenize(clear[i])]

#%% 清除停止詞
stopwords = ['rt']
stopwords.extend(stop_word) 
#stopwords.extend(number_1) 
stop_set = set(stopwords)


for  i in range(len(train['st'])) :
    for j in train['st'][i] :
        if j in stop_set:
            train['st'][i].remove(j)

#%%  變成字串
#train['tweet'] = train['st']
#train = train.drop("st", axis = 1)
train['st_str'] = train['st']
for i in range(len(train['st'])):
    train['st_str'][i] = ' '.join(train['st'][i])
#%%
#x, y = train['tweet'], train['class']
x, y = train['st_str'], train['class']
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    stratify=y)

my_tags = ['0','1','2']

#%% Naive Bayes Classifier for Multinomial Models

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score



nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)
test_ans = nb.predict(test['tweet'])

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
#%% Linear Support Vector Machine

from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)


y_pred = sgd.predict(X_test)
test_ans = sgd.predict(test['tweet'])
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
#%% Logistic Regression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


x, y = train['tweet'], train['class']
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y)
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
test_ans = logreg.predict(test['tweet'])
sample_upload['class'] = test_ans
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))
#%%





#df = train

#train_size = int(len(df) * .7)
#train_posts = df['st_str'][:train_size]
#train_tags = df['class'][:train_size]

#test_posts = df['st_str'][train_size:]
#test_tags = df['class'][train_size:]

x, y = train['tweet'], train['class']
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y)


max_words = 3000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(X_train) # only fit on train

x_train = tokenize.texts_to_matrix(X_train)
x_test = tokenize.texts_to_matrix(X_test)
x_ans =tokenize.texts_to_matrix(test['tweet'])


#encoder = LabelEncoder()
#encoder.fit(y_train)
#y_train = encoder.transform(y_train)
#y_test = encoder.transform(y_test)

num_classes = np.max(y_train) +1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
#y_ans = utils.to_categorical(sample_upload['class'], num_classes)

batch_size = 64
epochs = 4

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


#LOSS (1): sparse_categorical_crossentropy
#     (2): categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])
#print(classification_report(x_test, y_test,target_names=my_tags))
y_ans = model.predict_classes(x_ans)
sample_upload['class'] = y_ans
#%%
from keras.models import Sequential
from keras import layers
embedding_dim = 100

num_classes = np.max(y_train) +1

max_words = 1000
model = Sequential()
model.add(layers.Embedding(num_classes, embedding_dim, input_length=max_words))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    batch_size=10)
#%% 
sample_upload.to_csv("sample_upload.csv",index=False)
#%%
import pandas as pd
path = "C:/Users/rajat/Desktop/dataset/sentiment labelled sentences"
filepath_dict = {‘yelp’: ‘sentiment labelled sentences/            
                  yelp_labelled.txt’ ,‘amazon’: ‘sentiment labelled 
                  sentences /amazon_cells_labelled.txt’,‘imdb’: 
                  ‘sentiment labelled sentences/imdb_labelled.txt’}
df_list = []
for source, filepath in filepath_dict.items():
   df = pd.read_csv(filepath, names=[‘sentence’, ‘label’], sep=’\t’)
   # Add another column filled with the source name
   df[‘source’] = source 
   df_list.append(df)
df = pd.concat(df_list)
print(df.head())