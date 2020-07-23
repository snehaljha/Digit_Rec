import numpy as np
import matplotlib.pyplot as plt


import idx2numpy as ir

x_train = ir.convert_from_file('data/train-images.idx3-ubyte')

y_train = ir.convert_from_file('data/train-labels.idx1-ubyte')

x_test = ir.convert_from_file('data/t10k-images.idx3-ubyte')
y_test = ir.convert_from_file('data/t10k-labels.idx1-ubyte')

from sklearn.preprocessing import OneHotEncoder as OHE
ohe = OHE()
y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = ohe.transform(y_test.reshape(-1,1)).toarray()

x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

clas = Sequential()
clas.add(Conv2D(28, (3,3), input_shape=(28,28,1), activation='relu'))
clas.add(MaxPooling2D((2,2)))
clas.add(Flatten())
clas.add(Dense(units=56, activation='relu'))
clas.add(Dense(units=10, activation='sigmoid'))

clas.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  
clas.fit(x_train, y_train, batch_size=50, epochs=50)
clas.summary()
clas.evaluate(x_test, y_test)

import tensorflow as tf
import keras

keras.models.save_model(clas, 'keras_digit_clas.h5')

model = tf.keras.models.load_model('keras_digit_clas.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open('digit_clas_quantized.tflite', 'wb').write(tflite_model)
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_tensor_index = interpreter.get_input_details()[0]["index"]
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

test_image = np.expand_dims(x_test[1], axis=0).astype(np.float32)
interpreter.set_tensor(input_tensor_index, test_image)

interpreter.invoke()
interpreter.tensor(interpreter.get_output_details())()