import os
import cv2
import numpy as py
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist #importing the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1) #normalizing training and testing data
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential() #model
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #flat layer, takes 2d array and puts it into a 1d layer. Input layer
model.add(tf.keras.layers.Dense(128, activation = "relu")) #hidden layers
model.add(tf.keras.layers.Dense(128, activation = "relu"))
model.add(tf.keras.layers.Dense(10, activation = "softmax")) #output layer

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 3)

model.save("./model.keras")