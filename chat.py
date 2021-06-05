# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:02:06 2021

@author: dncna

"""

import random
import pyttsx3
import playsound
import numpy as np
from training import model_training
from tensorflow.keras.models import load_model
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def bag_of_words(s):
    
    bag = [0 for i in range(len(words))]
    
    tokenized_words = nltk.word_tokenize(s)
    tokenized_words = [stemmer.stem(w.lower()) for w in tokenized_words]
    
    for i in tokenized_words:
        for j, w in enumerate(words):
            if w==i:
                bag[j] = 1

    return np.array(bag)


def speak(text):
    print("Zira: " + text)
    engine.say(text)
    engine.runAndWait()
    

def chat():
     
    while True:
        inpu = input("You: ")
        if inpu.lower() == "quit":
            break
        
        results = model.predict(np.array([bag_of_words(inpu)])) 
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        for tg in data["data"]:
            if tg['tag'] == tag:
                respo = tg['responses']
                
        speak(random.choice(respo))


engine.say("Welcome to chatbot")
engine.say("Do you want to train me or chat with me ?")
engine.runAndWait()
    
w = input("Type 'Chat' to chat 'Train' to train: ")
    
if w.lower() == "train":
    array = model_training(True)
else:
    array = model_training()
    words = array[0]
    data = array[1]
    labels = array[2]
    model = load_model('./chat_bot.model')

    chat()