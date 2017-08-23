import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
import csv
import datetime
# Training parameters.
batch_size = 32
num_classes = 10
epochs = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

# The data, shuffled and split between train and test sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshapes data to 4D for Hierarchical RNN.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Converts class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

row, col, pixel = x_train.shape[1:]

# 4D input.
x = Input(shape=(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Final predictions and model.
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

class MyCallback(keras.callbacks.Callback):
    field = ['acc', 'loss']
    def on_batch_end(self, batch, logs):
        if(batch==0):
            with open('/logs/log{}.csv'.format(str(datetime.date.today())), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.field)
                writer.writerow([logs['acc'], logs['loss']])
            f.close()
        else:
            with open('/logs/log{}.csv'.format(str(datetime.date.today())), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([logs['acc'], logs['loss']])
            f.close()

callback_1 = keras.callbacks.TensorBoard(
    log_dir='../vis/tensorboard/test', 
    histogram_freq=1, 
    batch_size=32, 
    write_graph=True, 
    write_grads=False,
    write_images=False, 
    embeddings_freq=1, 
    embeddings_layer_names=None, 
    embeddings_metadata=None)

# callback_2 = MyCallback()
# Training.
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[callback_1])