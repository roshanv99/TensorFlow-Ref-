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
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])
model.compile(optimizer = "adam",loss = 'sparse_categorical_crossentropy',metrics = ["accuracy"])
model.fit(train_images,train_labels,epochs=5)
test_loss, test_acc = model.evaluate(test_images,test_labels)
print("Tested ACC: ", test_acc )

'''
The prediction function takes an array of inputs.
For each input, for this particular instance, we get 10 outputs, each suggesting how probable it is for it to be the output.
'''
prediction = model.predict(test_images)
print(prediction[0])
print(np.argmax(prediction[0]))
print(class_names[np.argmax(prediction[0])])
