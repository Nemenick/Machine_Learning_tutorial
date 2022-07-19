import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Dense, Flatten, Conv2D
# from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas
"""
Create a `Sequential` model and add a Dense layer as the first layer.

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
            # Now the model will take as input arrays of shape (None, 16)
            # and output arrays of shape (None, 32).
            # Note that after the first layer, you don't need to specify
            # the size of the input anymore:
    model.add(tf.keras.layers.Dense(32))
    model.output_shape
"""
# Classifico pari vs dispari

print("TensorFlow version:", tf.__version__)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape, type(x_train))
# plt.matshow(x_test[3])
# plt.show()
(y_train, y_test) = (y_train % 2, y_test % 2)

(x_train, x_test) = (x_train.reshape(60000, 28*28), x_test.reshape(len(x_test), 28*28))
print(x_train.shape, x_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=(784,), activation="relu"),
    # 16 è la dim dell'output, prende in input un vettore di 784 length, ma non stabilisco quanti vettori gli do
    tf.keras.layers.Dense(1, activation="sigmoid")
    # non ho bisgno di specificare più input
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

storia = model.fit(x_train, y_train, batch_size=32, epochs=1)

print(storia.history)

predizione = np.array(model.predict(x_test))
predizione = predizione.reshape(len(predizione),)
print("predict", predizione)
dizionario = {"predizione": predizione, "test_label": y_test}
print(dizionario)
dizionario = pandas.DataFrame.from_dict(dizionario)
dizionario.to_csv("Eccolo.csv")

