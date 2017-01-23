
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Reshape, Flatten, Dropout
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import TensorBoard, CSVLogger

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

x_train / 255
x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()

conv_layer = Convolution2D(32, 5, 5, border_mode='same', input_shape=(1, 28, 28))
model.add(conv_layer)

pool_layer1 = MaxPooling2D(
    pool_size = (2, 2),
    strides = None,
    border_mode = 'same',
    dim_ordering = 'default'
)
model.add(pool_layer1)

conv_layer2 = Convolution2D(64, 5, 5, border_mode='same')
model.add(conv_layer2)

pool_layer2 = MaxPooling2D(
    pool_size = (2, 2),
    strides = None,
    border_mode = 'same',
    dim_ordering = 'default'
)
model.add(pool_layer2)
model.add(Dropout(.5))
model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Activation('softmax'))
# model.add(Activation('softmax'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

TensorBoard(
    log_dir = './tmp/mnist',
    histogram_freq = 0,
    write_graph = True,
    write_images = False
)

CSVLogger(
    './tmp/mnist/mnist.csv'
)

# weights = model.load_weights('tmp/mnist', by_name=False)
# model.set_weights(weights)

model.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    nb_epoch = 10,
    batch_size = 50,
    verbose = 2
)
from keras.utils.visualize_util import plot
plot(
    model,
    to_file='model.png',
    show_shapes=True,
)

# model.save_weights('tmp/mnist')

# scores = model.evaluate(
#     x_test,
#     y_test,
#     verbose = 2
# )
# print('Classification Error: %.2f%%' % 100- scores[1] * 100)
