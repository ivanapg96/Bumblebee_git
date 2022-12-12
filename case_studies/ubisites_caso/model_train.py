# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:08:44 2019
@author: dell
"""
import random

import numpy as np
from Bio import SeqIO
import re
from nltk import trigrams, bigrams
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Convolution1D, Dropout, Activation, MaxPooling1D
from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import os
import tensorflow
from word_embbeding_main import WordEmbedding
import numpy
from gensim.models import Word2Vec

np.set_printoptions(threshold=np.inf)

os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
print(gpus)
tensorflow.debugging.set_log_device_placement(True)
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#reportname = 'one-hot'
reportname = 'bumblebee_w_es_rlr_batch_2'
#reportname = 'proposed_w_es_rlr_embedding'
#reportname = 'bumblebee_proposed_embedding'
#reportname = 'proposed_embedding'
seed = 10
np.random.seed(seed)
tensorflow.random.set_seed(seed)
random.seed(seed)
tensorflow.debugging.set_log_device_placement(True)
strategy = tensorflow.distribute.MirroredStrategy()


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

from tensorflow.keras.utils import to_categorical
def hot_encoded_sequence(sequences):
    sequences_integer_ecoded = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
    for seq in sequences:
        # define a mapping of chars to integers

        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in seq]
        sequences_integer_ecoded.append(integer_encoded)

    # one hot encoding
    shape_hot = len(alphabet) * len(sequences[0])  # 20000
    encoded = to_categorical(sequences_integer_ecoded) # shape (samples, 1000,20)
    fps_x = encoded.reshape(encoded.shape[0], shape_hot)  # shape (samples, 20000)
    return fps_x , sequences_integer_ecoded


texts = []
for index, record in enumerate(SeqIO.parse('/home/igomes/Bumblebee/ubisites_caso/datasets/trainset1.fasta', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        # print(item),
        temp_str = temp_str + " " + item[0] + item[1]
        # temp_str = temp_str + " " +item[0]
    texts.append(temp_str)
sequences = []
for index, record in enumerate(SeqIO.parse('/home/igomes/Bumblebee/ubisites_caso/datasets/trainset1.fasta', 'fasta')):
    sequences.append(record.seq)
seq = []
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
    doc = re.sub(stop, '', doc)
    seq.append(doc.split())

# w2v_model = Word2Vec.load('word2vec.model')
# embedding_matrix = w2v_model.wv.vectors
# vocab_list = list(w2v_model.wv.vocab.keys())
#
w2v_model = WordEmbedding(ngram_len=2, emb_matrix_file='trained_word_embeddings/word2vecmodel_skipgram_matrix.csv')
embedding = w2v_model.get_emb_matrix()
vocab_list = list(embedding.keys())
embedding_matrix = []
for i in embedding.values():
    embedding_matrix.append(list(i))
embedding_matrix = numpy.array([numpy.array(xi) for xi in embedding_matrix])


word_index = {word: index for index, word in enumerate(vocab_list)}


def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence


X_data = np.array(list(map(get_index, seq))) #original
#X_data , encoded = hot_encoded_sequence(sequences)

Y_data = np.load("/home/igomes/Bumblebee/ubisites_caso/datasets/y_trainset.npy")
print(X_data.shape)
#
maxlen = 30
with strategy.scope():
    model = Sequential()
    model.add(Embedding( #original
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False))
    #model.add(Embedding(input_dim=20 , output_dim= 20 , input_length=31)) #one.hot
    model.add(Convolution1D(filters=128, kernel_size=3))
    #model.add(Convolution1D(filters=128, kernel_size=3, input_shape=(30, 20))) #original
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Convolution1D(filters=128, kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Convolution1D(filters=128, kernel_size=3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    #
    model.add(Flatten())
    #
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    # model.add(Convolution1D(filters=64, kernel_size=6, input_shape=(30, 20)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    #
    # model.add(Convolution1D(filters=16, kernel_size=4))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    # #
    # model.add(Convolution1D(filters=4, kernel_size=2))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    #
    # model.add(Dense(8))
    # model.add(Activation('relu'))
    # model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',specificity,recall])


    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0) #original

#one-hot
#encoded = np.array(encoded)
#X, X_test, Y, Y_test = train_test_split(encoded, Y_data, test_size=0.2, random_state=0)

#embedding
print(X_data.shape)
X, X_test, Y, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

# simple early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=50)
# reduce learning rate reduce_lr_factor=0.5
reduce_lr_factor=0.2
reduce_lr_min=0.00001
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor,
                              patience=20, min_lr=reduce_lr_min, verbose=1)
# checkpoint
path = '/home/igomes/Bumblebee/ubisites_caso/datasets'
filepath = os.path.join(path, 'weights-{{epoch:02d}}-{{val_loss:.2f}}.hdf5')
cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0,
                     save_best_only=True, save_weights_only=False,
                     mode='auto', save_freq='epoch', period=1)
epochs = 120

batch_size = 2024+24
patience_es = 50
patience_lr = 20
# history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=1,
#                     shuffle=True)
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=1,
                    shuffle=True, callbacks=[es, reduce_lr, cp])
model.save( reportname +'.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy' + reportname)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(reportname + ".png")



sequences_test = []
for index, record in enumerate(SeqIO.parse('/home/igomes/Bumblebee/ubisites_caso/datasets/testset1.fasta', 'fasta')):
    sequences_test.append(record.seq)

# texts = []
# for index, record in enumerate(SeqIO.parse('/home/igomes/Bumblebee/ubisites_caso/datasets/testset1.fasta', 'fasta')):
#     tri_tokens = bigrams(record.seq)
#     temp_str = ""
#     for item in ((tri_tokens)):
#         #print(item),
#         temp_str = temp_str + " " +item[0] + item[1]
#         #temp_str = temp_str + " " +item[0]
#     texts.append(temp_str)
# #
# seq=[]
# stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
# for doc in texts:
#     doc = re.sub(stop, '', doc)
#     seq.append(doc.split())
#
# #
# word_index = {word: index for index, word in enumerate(vocab_list)}
# #
# def get_index(sentence):
#     global word_index
#     sequence = []
#     for word in sentence:
#         try:
#             sequence.append(word_index[word])
#         except KeyError:
#             pass
#     return sequence
# #
#
# X_data_test= np.array(list(map(get_index, seq)))
# _ , X_data_test = hot_encoded_sequence(sequences_test)
# Y_data_test = np.load("/home/igomes/Bumblebee/ubisites_caso/datasets/y_testset.npy")
X_data_test = X_val
Y_data_test = Y_val
#
model = load_model(reportname+'.h5')


#SCORES
y_prob = model.predict(X_test, batch_size = 50)
y_pred = []
for i in y_prob:
    list2 = []
    for x in i:
        list2.append(int(round(x, 0)))
    y_pred.append(list2)

test_auc = roc_auc_score(Y_test, y_prob)
print("test_auc: ", test_auc)
cm = confusion_matrix(Y_test, y_pred)
print(cm)
#####################################

scores = {}


scores['Accuracy'] = accuracy_score(Y_test, y_pred)
scores['MCC'] = matthews_corrcoef(Y_test, y_pred)
scores['log_loss'] = log_loss(Y_test, y_prob)


# binary
scores['f1 score'] = f1_score(Y_test, y_pred)
scores['roc_auc'] = roc_auc_score(Y_test, y_pred)
precision, recall, thresholds = precision_recall_curve(Y_test, y_pred)
scores['Precision'] = precision
scores['Recall'] = recall
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
# scores['fdr'] = float(fp) / (tp + fp)
scores['sn'] = float(tp) / (tp + fn)
scores['sp'] = float(tn) / (tn + fp)


acc = accuracy_score(Y_test, y_pred)
mcc = matthews_corrcoef(Y_test, y_pred)
print('accuracy', acc)
print('MCC', mcc)

report = classification_report(Y_test, y_pred, output_dict=False)
print(report)
w = open( reportname+"_results.txt", "a")
model.summary(print_fn=lambda x: w.write(x + '\n'))
w.writelines('epochs: ' + str(epochs))
w.write('\n')
w.writelines( 'batch_size: ' + str(batch_size))
w.write('\n')

if es:
    w.writelines('EarlyStopping patience: ' + str(patience_es))
    w.write('\n')
    w.writelines('Reduce Learning Rate patience: ' + str(patience_lr))
    w.write('\n')
    w.writelines('rlr min: '+ str(reduce_lr_min))
    w.write('\n')
    w.writelines('rlr factor: ' + str(reduce_lr_factor))
    w.write('\n')
w.write('Metrics for Independent Test')
w.write('\n')

for j , i in scores.items():
    w.write(str(j) + ',')
    w.write(str(i) + ',')
    w.write('\n')

w.write('confusion matrix: ' + str(cm))
w.close()
#model = load_model('model_bumblebee.h5')

#preds = model.predict_classes(X_data)

#preds = model.predict_classes(X_data)

# scores = {}
#
# try:
#     y_prob = model.predict_proba(X_data_test)
#     y_pred = model.predict(X_data_test)
#     y_pred = np.argmax(y_prob, axis=1)
#
# except:
#     y_prob = model.predict(X_data_test)
#     y_pred= np.argmax(y_prob,axis=1)
#
#
#
# scores['Accuracy'] = accuracy_score(Y_data_test, y_pred)
# scores['MCC'] = matthews_corrcoef(Y_data_test, y_pred)
# scores['log_loss'] = log_loss(Y_data_test, y_prob)
#
#
# # binary
# scores['f1 score'] = f1_score(Y_data_test, y_pred)
# scores['roc_auc'] = roc_auc_score(Y_data_test, y_pred)
# precision, recall, thresholds = precision_recall_curve(Y_data_test, y_pred)
# scores['Precision'] = precision
# scores['Recall'] = recall
# tn, fp, fn, tp = confusion_matrix(Y_data_test, y_pred).ravel()
# # scores['fdr'] = float(fp) / (tp + fp)
# scores['sn'] = float(tp) / (tp + fn)
# scores['sp'] = float(tn) / (tn + fp)
#
#
# acc = accuracy_score(Y_data_test, y_pred)
# mcc = matthews_corrcoef(Y_data_test, y_pred)
# print('accuracy', acc)
# print('MCC', mcc)
#
# report = classification_report(Y_data_test, y_pred, output_dict=False)
# cm = confusion_matrix(Y_data_test, y_pred)
# print(cm)
# print(report)