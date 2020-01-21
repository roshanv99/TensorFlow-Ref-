import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#from keras.preprocessing import sequence


data = keras.datasets.imdb
(train_data, train_labels),(test_data,test_labels) = data.load_data(num_words=10000)
#Since we'll be comparing movie reviews, we will only be using the 10,000 most recurring words

word_index = data.get_word_index()
#The data is basically a tuple of intergers that map to a word
word_index = {k:(v+3) for k,v in word_index.items()}


word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

#We are going to preprocess our data to make them equalto the same length. So we're just add characters to 250 or remove characters greater than 250
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value =word_index["<PAD>"], padding = "post", maxlen = 250)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value =word_index["<PAD>"], padding = "post", maxlen = 250)

def decode_review (text):
    return " ".join([reverse_word_index.get(i,'?') for i in text])

print(decode_review(test_data[0]))

#Model:
model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation = 'relu'))
model.add(keras.layers.Dense(1,activation = 'sigmoid'))
'''
We want our data to give us a probability of how good or bad it is. So the sigmoid function will give us an output in the range of 0 to 1
'''
model.summary()


#Training out data:
model.compile(optimizer='adam', loss = "binary_crossentropy",metric = ["accuracy"])
#We will split our data into 2 sets: Validation
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train,y_train,epochs = 40, batch_size = 512,validation_data = (x_val,y_val), verbose = 1 )
#Batch Size is total number of movie reviews we will load in at once

results = model.evaluate(test_data,test_labels)
print(results)

#To save the model we simply save it:
model.save("model_name.h5")

'''
#To load the model:
model = keras.model.load_model("model_name.h5")

#It is a good practice to make multiple model with varying pararmeters overnight and save only the best once

''' 
