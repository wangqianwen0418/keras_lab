'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
from keras.preprocessing import sequence
import numpy as np
from keras.models import Model
from keras import layers
from keras.datasets import imdb

max_features = 20000 # only count the 20000 most frequent words
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')

# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)

print('Build model...')
in_layer = layers.Input(shape=(None,), name="a")
x = layers.Embedding(max_features, 128, name="b")(in_layer)
x= layers.Masking()(x)
x = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2, name="c")(x)
out_layer = layers.Dense(1, activation='sigmoid', name="d")(x)
model = Model(inputs=[in_layer], outputs=[out_layer])
# print(model.summary())
# for layer in model.layers:
#     try:
#         print(layer.name, layer.count_params())
#     except Exception:
#         print(layer.name)

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
for seq, label in zip(x_train, y_train):
       model.train_on_batch(np.array([seq]), [label])
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=15,
#           validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)