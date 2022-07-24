from tensorflow import keras
model = keras.models.load_model('MNIST_conv.hdf5')

print(model.summary())