# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:03:50 2019

@author: wei
"""
#%%
import re
import pandas as pd
from sklearn.model_selection import train_test_split



#%%  
def clean_en_text(ce):
    comp = re.compile("[^A-Z^a-z^0-9^ ^']")
    return comp.sub('', ce)

def clean_stop_word(sw):
    list_words=word_tokenize(sw)
    sw=[w for w in list_words if not w in list_stopWords]
    for w in list_words :
        if not w in stop_word
    

#   [w for w in text if not w in stop_word]
    returnx

#%%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
op = open('stop.txt', "r")
stop_word = op.readlines()
op.close()
#%% test
x = train['tweet'].to_list()
sw = x[0]
sw = clean_en_text(sw)
list_words=word_tokenize(sw)
sw = [w for w in list_words if not w in stop_word]

#%% 資料預處理
#去除符號
x = train['tweet'].to_list()
x = x[0]
print(x)
x = clean_en_text(x)
print(x)
x = clean_stop_word(x)
print(x)
#%%
x = train['tweet'].to_list()
for i in range(len(x)) :
    x[i] = clean_en_text(x[i])
    x[i] = clean_stop_word(x[i])
    
train['tweet'] = x

x = test['tweet'].to_list()
for i in range(len(x)) :
    x[i] = clean_en_text(x[i])
    
test['tweet'] = x

print(train['tweet'][0])

#filtered_words = [word for word in word_list if word not in stopwords.words('english')]
remainderWords = list(filter(lambda a: a not in stop_word and a != '\n', train['tweet']))
print(train['tweet'][0])
#%% ngrams
s = "Natural-language processing (NLP) is an area of computer science " \
    "and artificial intelligence concerned with the interactions " \
    "between computers and human (natural) languages."

from nltk.util import ngrams

s = s.lower()
s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
tokens = [token for token in s.split(" ") if token != ""]
output = list(ngrams(tokens, 5))
output = list(ngrams(train['st'], 5))
#%%CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
 
#语料
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

t = train['tweet'].to_list()
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
#X = vectorizer.fit_transform(train['tweet'])
#X = vectorizer.fit_transform(corpus)
X = vectorizer.fit_transform(t)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print (word)
#查看词频结果
print (X.toarray())
train['CV'] = X.toarray().tolidt()
#%% st單字拆解

clear = train['st']
for  i in range(len(clear)) :
    clear[i] = [t for t in word_tokenize(clear[i])]

#%% 全部轉小寫
for i in range(len(train['st'])):
    for j in range(len(train['st'][i])):
        train['st'][i][j] = str.lower(train['st'][i][j])
#%% 列出所有組合
list_all = []
list_all = train['st'].to_list()
list_aw = []
for i in range(len(list_all)):
    for j in range(len(list_all[i])):
        list_aw.append(list_all[i][j])
sort = sorted(set(list_aw))
number_1 = []
for i in sort:
    if list_aw.count(i) == 1:
        number_1.append(i)
#%% 清除停止詞
stopwords = ['rt']
stopwords.extend(stop_word) 
#stopwords.extend(number_1) 
stop_set = set(stopwords)


for  i in range(len(train['st'])) :
    for j in train['st'][i] :
        if j in stop_set:
            train['st'][i].remove(j)
#%%
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(len(macronum), activation='softmax')(l_dense)












