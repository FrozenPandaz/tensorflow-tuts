from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.callbacks import TensorBoard, CSVLogger
import numpy as np

x_train = np.random.rand(100).astype(np.float32)

y_train = x_train**2 * 3 + 2

# y_train = np.vectorize(lambda y: y)(y_train)
x_train = [x_train]
y_train = [y_train]
print(x_train)

model = Sequential()

# model.add(Flatten(input_shape=(100, 1)))

# model.add(Reshape(input_shape=(100, 1), target_shape=(100, 1, 1)))
layer_1 = Dense(2, input_dim=1)
model.add(layer_1)

model.add(Activation('relu'))

model.add(Dense(2, input_dim=1))

model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('softmax'))

# model.add(Flatten())

# model.add(Activation('relu'))

model.summary()

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    nb_epoch=4,
    batch_size=20
)

# print(model.save_weights('./tmp/keras'))from IPython.display import SVG
from keras.utils.visualize_util import plot
plot(
    model,
    to_file='model.png',
    show_shapes=True,
)

# score = model.predict([2])
