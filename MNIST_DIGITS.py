import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical
import numpy as np

"""
def categorical(vettore):
    y_test1 = to_categorical(vettore)
    y_test2 = [[vettore[j] == i for i in range(10)] for j in range(len(vettore))]
    print("1\n", y_test1)
    print("1\n", np.array(y_test2))
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(y_train, ytest) = (to_categorical(y_train), to_categorical(y_test))

(x_train, x_test) = (x_train.reshape(60000, 28*28), x_test.reshape(len(x_test), 28*28))
model = keras.models.Sequential([
    Dense(64, activation="relu"),
    Dense(10, activation="relu")
])

print(model.predict(x_test))