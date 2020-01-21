import tensorflow as tf
from tensorflow import keras
#Keras is an API for tensor flow
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,test_labels) = data.load_data()

#This specific data will have 10 labels ie each image will have a specific label assigned to it

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0
#We divide every value of the numpy array by 255 to get the values in the range of 0 to 1
