# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 21:39:46 2021

@author: dilumika

"""

import random
import json
import pickle
import numpy as np

import nltk

#If it is the very first time of using nltk, you need to run this too
#nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential

def model_training(train = False):
    with open('intents.json') as file:
        data = json.load(file)
        
    words = []
    labels = []
    X = []
    Y = []
    
    for i in data['data']:
        for j in i['patterns']:
            wrds = nltk.word_tokenize(j)
            words.extend(wrds)
            X.append(wrds)
            Y.append(i['tag'])
            
        if i['tag'] not in labels:
            labels.append(i['tag'])
                
    
    words = [stemmer.stem(word.lower()) for word in words if word!='?']
    words = sorted(list(set(words)))
    labels = sorted(labels)
    
    if train:
    
        training = []
        output = []
        
        
        out_empty = [0 for _ in range(len(labels))]
        
        for x, doc in enumerate(X):
            bag = []
            
            wrds = [stemmer.stem(w.lower()) for w in doc]
            
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
                    
                    
            output_row = out_empty[:]
            output_row[labels.index(Y[x])] = 1
            
            training.append(bag)
            output.append(output_row)
            
        training = np.array(training)
        output = np.array(output)
        
        
        model = Sequential()
        
        model.add(Dense(128,activation='relu', input_shape=(len(training[0]), )))
        model.add(Dropout(0.5))
        
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(32,activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(len(output[0]),activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()
        
        model.fit(training, output, epochs=2000, batch_size=5)
        model.save('chat_bot.model')


    return([words, data, labels])