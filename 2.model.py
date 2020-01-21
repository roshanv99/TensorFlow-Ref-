'''
Our input layer is going to be the flattened array of all the pixels in the image.
Flattening the array is basically making the multidimentional array into a linear array.
Out output is going to be a value between 0-9 (because we have 9 labels)

If the size of the image is 28x28 then,
    1. The total number of inputs will be 28 * 28 = 784
    2. The total number of weights and biases will be 784*10 = 7840 (labels)


'''
import tensorflow as tf
from tensorflow import keras
#Keras is an API for tensor flow
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,test_labels) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images/255.0
test_images = test_images/255.0

'''
We need to first create the architecture of our model
    1. We first flatten our model
    2. We create a hidden layer in which every input is connected to each neuron (relu: Rectified Linear Unit)
    3. We create another hidden layer of size 10 in which the probablity will add upto 1
'''
model = keras.Sequential([
#Flattening a tensor means to remove all of the dimensions except for one
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

'''
We now create parameters for our model
'''
model.compile(optimizer = "adam",loss = 'sparse_categorical_crossentropy',metrics = ["accuracy"])

#Now we train our model
#epochs is the number of times our model will see the same image.
model.fit(train_images,train_labels,epochs=5)

#Now we test our images:
test_loss, test_acc = model.evaluate(test_images,test_labels)

print("Tested ACC: ", test_acc )
