import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#from keras.preprocessing import sequence


data = keras.datasets.imdb
(train_data, train_labels),(test_data,test_labels) = data.load_data(num_words=88000)
#Since we'll be comparing movie reviews, we will only be using the 10,000 most recurring words

word_index = data.get_word_index()
#The data is basically a tuple of intergers that map to a word
word_index = {k:(v+3) for k,v in word_index.items()}


word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

def review_encode(s):
    encoded = [1]
    for w in s:
        if w in word_index:
            encoded.append(word_index[w.lower()])
        else:
            encoded.append(2)
    return encoded


model = keras.models.load_model("model_name.h5")
with open("text.txt",encoding = 'utf-8') as f:
    for line in f.readlines():
        #We replace words that need no mapping
        nline = line.replace(',','').replace('.','').replace('(','').replace(')','').replace(':','').replace('\"','').strip()
        #We encode the data to only be 250 words
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value =word_index["<PAD>"], padding = "post", maxlen = 250)

        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
