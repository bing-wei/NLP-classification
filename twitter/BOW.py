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
#from nltk.corpus import wordnet as wn
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
#from nltk import pos_tag
#from nltk.corpus import stopwords


train = pd.read_csv('C://Users//wei//Desktop//python//twitter//train.csv')
test = pd.read_csv('C://Users//wei//Desktop//python//twitter//test.csv')
sample_upload = pd.read_csv('C://Users//wei//Desktop//python//twitter//sample_upload.csv')
#%%

# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
train['tweet'] = [entry.lower() for entry in train['tweet']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
train['tweet']= [word_tokenize(entry) for entry in train['tweet']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(train['tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    train.loc[index,'text_final'] = str(Final_words)


# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
test['tweet'] = [entry.lower() for entry in test['tweet']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
test['tweet']= [word_tokenize(entry) for entry in test['tweet']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(test['tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    test.loc[index,'text_final'] = str(Final_words)

#class_0 = train[train['class'] == 0 ]
#%%
# 原本 1000
from sklearn.utils import class_weight
#x, y = train['text_final'], train['class']
x, y = train['tweet'], train['class']
X_train, X_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    stratify=y)



max_words = 26183
tokenize = text.Tokenizer(num_words=None, char_level=False)
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
epochs = 20




model = Sequential()
model.add(Dense(4096, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))   #防止overfitting
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
#%%
'''Access the loss and accuracy in every epoch'''
loss_ce	= history.history.get('loss')
acc_ce 	= history.history.get('val_accuracy')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss_ce)), loss_ce,label='CE')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_ce)), acc_ce,label='CE')
plt.title('Accuracy')
#plt.savefig('00_firstModel.png',dpi=300,format='png')
plt.show()
#plt.close()
#print('Result saved into 00_firstModel.png')
#%% 預測答案及更改答案
x_ans =tokenize.texts_to_matrix(test['text_final'])
#y_ans = utils.to_categorical(sample_upload['class'], num_classes)
y_ans = model.predict_classes(x_ans)
sample_upload['class'] = y_ans
sample_upload.to_csv("sample_upload.csv",index=False)

import os
os.getcwd()
