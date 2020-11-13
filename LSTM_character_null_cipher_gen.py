# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:58:49 2020

@author: 17013
"""
import time
import glob
import os
# this prevents keras from outputting GPU activity to console
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import keras
import tensorflow as tf
import numpy
import random
import matplotlib.pyplot as plt
from datetime import datetime
from os import listdir
from os.path import isfile, join


today = datetime.today()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

dirname = os.path.dirname(__file__)

# HYPER PARAMETERS
learning_rate = 0.015
epochs = 7
batch_size = 50
diversity = 1.0
# text splitting parameters
maxlen = 160
step = 3
# create dicts for int to char and vise versa so they are created without loading in text
# lower alpha, space, and a few basic punctuations
char_count = 35



# define the LSTM model -----------------------
model = Sequential()
model.add(Dense(200, input_shape=(maxlen , char_count)))
model.add(Bidirectional(LSTM(1000, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(1000, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(1000, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(1000)))
model.add(Dropout(0.5))
model.add(Dense(char_count, activation='softmax'))
optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ask if we want to load a model
load_files = []
load = input('would you like to load a model? leave blank to train new ')

if load:
    for c, filename in enumerate(glob.glob(os.path.join('./keras_training', '*.hdf5'))):
        print('filename: {} num: {}'.format(filename,  c))
        load_files.append(filename)
    file_to_load = int(input('What file to load? number '))

    # load the network weights
    model.load_weights(load_files[file_to_load])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_more = input('would you like to train more? ')

# we don't want to load a model, we must be training a new one, or training an old one
if not load or train_more:
    # GET DATA FROM FILES
    files = []

    for c, filename in enumerate(glob.glob(os.path.join('./text_data', '*.txt'))):
        size = os.stat(filename).st_size

        print('filename: {} size: {} kB num: {}'.format(filename, size//1000, c))
        files.append(filename)

    file_num = int(input('What file would you like to use? enter number of file '))

    start = time.time()

    with open(files[file_num], 'r', encoding='utf-8') as f: # open in readonly mode
        my_string = f.read().lower().replace('\n', ' ')

        # replace left and right double quotations with nondirectional double quotes
        my_string = re.sub(r'[“”]', '"', my_string)

        my_string = ' '.join(my_string.split())
        # remove all characters that are not lower alpha or punc included in re
        my_string = re.sub(r'[^a-z \-!.?,;\"\']', '', my_string)

        # create mapping of unique chars to integers
        chars = sorted(list(set(my_string)))

        n_chars = len(my_string)
        n_vocab = len(chars)
        print("Total Characters: ", n_chars)
        print("Total Vocab: ", n_vocab)

        # this is to allow traning on works that don't include certain characters, mainly quotations
        symbols = ["'", '"']
        if len(chars) != 35:
            for symbol in ["'", '"']:
                if symbol not in chars:
                    chars.append(symbol)

        char_to_int = dict((c, i) for i, c in enumerate(chars))
        int_to_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(my_string) - maxlen, step):
            sentences.append(my_string[i : i + maxlen])
            next_chars.append(my_string[i + maxlen])
        print("Number of sequences:", len(sentences))

        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.int)
        y = np.zeros((len(sentences), len(chars)), dtype=np.int)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_to_int[char]] = 1
            y[i, char_to_int[next_chars[i]]] = 1

    onlyfiles = [f for f in listdir(dirname+'\\text_data') if isfile(join(dirname+'\\text_data', f))]
    # define the checkpoint
    filepath="./keras_training/"+onlyfiles[file_num][:-4]+"-{val_accuracy:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    reduce_alpha = ReduceLROnPlateau(monitor ='loss', factor = 0.2, patience = 1, min_lr = 0.001)
    callbacks_list = [checkpoint, reduce_alpha]

    loss_his = []
    val_loss_his = []
    acc_his = []
    val_acc_his = []

    for epoch in range(epochs):
        this_history = model.fit(x, y, validation_split=0.4, epochs=1, batch_size=batch_size, callbacks=callbacks_list)

        loss_his.append(this_history.history['loss'])
        val_loss_his.append(this_history.history['val_loss'])
        acc_his.append(this_history.history['accuracy'])
        val_acc_his.append(this_history.history['val_accuracy'])

        if epoch % 4 == 0:

            print()
            print("Generating text after epoch: %d" % epoch)

            start_index = random.randint(0, len(x) - maxlen - 1)
            for diversity in [0.8, 1.0, 1.2]:
                print("...Diversity:", diversity)

                generated = ""
                sentence = my_string[start_index : start_index + maxlen]
                print('...Generating with seed: "' + sentence + '"')

                for i in range(300):
                    x_pred = np.zeros((1, maxlen, len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, char_to_int[char]] = 1.0

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)
                    next_char = int_to_char[next_index]
                    sentence = sentence[1:] + next_char
                    generated += next_char

                print("...Generated: ", generated)
                print()

    # summarize history for accuracy
    plt.plot(acc_his)
    plt.plot(val_acc_his)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./plots/acc_{}'.format(today.strftime("%d-%m-%Y-%H-%M-%S")))
    plt.show()

    # summarize history for loss
    plt.plot(loss_his)
    plt.plot(val_loss_his)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./plots/loss_{}'.format(today.strftime("%d-%m-%Y-%H-%M-%S")))
    plt.show()

# GENERATE TEXT
seed_text = input('Enter the seed to use to start or leave blank for random seed. ').lower()
generated = ''
null_cipher = input('enter null cipher: ')
null_cipher = null_cipher.replace(' ', '')
null_len = len(null_cipher)
null_index = 0

if len(seed_text) < maxlen:
    sentence = seed_text.rjust(maxlen)

for i in range(200):
    x_pred = np.zeros((1, maxlen, char_count))

    for t, char in enumerate(sentence):
        x_pred[0, t, char_to_int[char]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = int_to_char[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char

    # if the next generated letter is a space, add the next letter in null_cipher
    if next_char == ' ':
        if null_index < null_len:
            next_char += null_cipher[null_index]
            null_index += 1
            sentence = sentence[2:] + next_char
            generated += next_char

print('seed: ', seed_text)
print('generated: ', generated)


end = time.time()
print('total time: {} mins {} seconds'.format((end - start) // 60, round((end - start) % 60, 1)))