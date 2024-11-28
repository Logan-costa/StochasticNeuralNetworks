import os
import cv2
import numpy as py
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model("./model.keras")

mnist = tf.keras.datasets.mnist #importing the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1) #normalizing training and testing data
x_test = tf.keras.utils.normalize(x_test, axis = 1)

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)