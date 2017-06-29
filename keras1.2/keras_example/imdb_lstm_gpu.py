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
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from profiler import profile
from model_util import make_model

#Result dictionary
global ret_dict
ret_dict = dict()

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model = make_model(model, loss='binary_crossentropy',
                                                optimizer='adam',
                                                metrics=['accuracy'])

print('Train...')
def train_func():
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
              validation_data=(X_test, y_test))
ret = profile(train_func)

ret_dict["training_time"] = str(ret[0]) + ' sec'
ret_dict["max_memory"] = str(ret[1]) + ' MB'

ret_dict["training_accuracy"] = model.evaluate(X_train, y_train, verbose=0)[1]
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
ret_dict["test_accuracy"] = acc
print('Test score:', score)
print('Test accuracy:', acc)
