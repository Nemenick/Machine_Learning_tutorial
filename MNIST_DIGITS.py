import tensorflow as tf
import numpy as np
import pandas
from tensorflow import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical


"""
def categorical(vettore):
    y_test1 = to_categorical(vettore)
    y_test2 = [[vettore[j] == i for i in range(10)] for j in range(len(vettore))]
    print("1\n", y_test1)
    print("1\n", np.array(y_test2))
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(y_train, y_test) = (to_categorical(y_train), to_categorical(y_test))

(x_train, x_test) = (x_train.reshape(60000, 28*28), x_test.reshape(len(x_test), 28*28))
model = keras.models.Sequential([
    Dense(64, activation="relu"),
    Dense(16, activation="relu"),            # se è meno di 10 NON VA! (BOOTLENECK)
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy'])

storia = model.fit(x_train, y_train, batch_size=32, epochs=10)

print(storia.history)

predizione = model.evaluate(x_test, y_test)
print(len(predizione), y_test.shape, type(predizione), type(y_test))
print("predict", predizione)
dizionario = {}
"""yt = [[0 for _ in range(len(y_test))] for __ in range(len(y_test[0]))]
pt = [[0 for j in range(len(predizione))] for i in range(len(predizione[0]))]

for i in range(len(y_test)):
    for j in range(len(y_test[0])):
        yt[j][i] = y_test[i][j]
        pt[j][i] = predizione[i][j]

for i in range(len(pt)):
    dizionario["pred_" + str(i)] = pt[i]
for i in range(len(yt)):
    dizionario["y_" + str(i)] = yt[i]
print(dizionario)
dizionario = pandas.DataFrame.from_dict(dizionario)
dizionario.to_csv("Eccolo.csv",index=False)"""


