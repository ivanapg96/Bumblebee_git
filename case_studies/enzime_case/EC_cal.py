import sys
print("User Current Version:-", sys.version)
import pickle
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import time
import functools
TF_FORCE_GPU_ALLOW_GROWTH= True
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus[:1], 'GPU')

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

from propythia.deep_ml import DeepML
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout, Bidirectional
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GRU, BatchNormalization, Masking
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef

from collections import Counter
from word_embedding_main import WordEmbedding as wv
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
# todo
# go check the notebooks for examples at https://github.com/BioSystemsUM/propythia
# examples and docs (pdf and quickstart)
#############################################################################################################
# 1 define X and Y and split into Test and Train
# train_test_split scikit learn


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def turn_single_label(column, data):
    l = []
    for ec_list in data[column]:
        ec_l = set(ec_list)
        l.append(ec_l)
    data['ec_single_label']=l
    data = data.loc[data['ec_single_label'].apply(len)<2,:]
    return data

def remove_zeros(column, data):
    list_zeros=[0, 0.0, '0', '0.0', '0.0.0', '0.0.0.0']
    l = []
    for ec_list in data[column]:
        ec_l = [x for x in ec_list if x not in (list_zeros)]
        l.append(ec_l)
    data['non_negative'] =l

    data = data.loc[data['non_negative'].apply(len)>0,:]
    return data

def get_ec_1_level(data, single_label=True):
    # get all until the 1 level (everything until the last dot    175805
    l = []
    for ec_list in data['ec_number']:
        ec_1 = [x.strip()[0] for x in ec_list.split(';') ]
        # [^,]* = as many non-dot characters as possible,
        # . = a dot
        l.append(list(set(ec_1)))
    data['ec_number1']=l
    if single_label:
        data = turn_single_label('ec_number1', data)
    else:
        pass
    data['ec_number1']= [xs[0] for xs in data['ec_number1']]
    counts = Counter(x for xs in data['ec_number1'] for x in set(xs))
    counts.most_common()
    df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    df_sorted = df.sort_values(by=[0], ascending=False)
    print(df_sorted)

def get_seqs_vectors(data, max_len: int = None , ngram: int = 3, ngram_selected =None ):
    w2vmodel = wv(emb_matrix_file='protVec_100d_3grams.csv')


    n=1
    print('seq to Vec')
    # b=len(w2vmodel.convert_seq2vec(method=2, sequence=data['sequence'][0]))
    k = []
    num_of_dim_of_vector = len(w2vmodel.convert_seq2vec(method=2, sequence=data['sequence'][0]))
    print(num_of_dim_of_vector)
    for i in range(num_of_dim_of_vector):
        k.append('dim' + str(i+1))
    for seq in data['sequence']:
        print(n, '/', len(data['sequence']))
        if max_len:
            seq = seq[0:max_len]
        seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
        seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
        seq3 = seq2.replace('U', 'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
        seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
        seq = seq4.replace('X', '')  # unknown character eliminated

        vector = w2vmodel.convert_seq2vec(method=2, sequence=seq, ngram_remove=ngram_selected)
        print(' sera?')
        g=0
        for j in vector:
            data[k[g]] = j
            g+=1

        n += 1

        return

#
def transform_seq(seq, max_len):
        if max_len:
            seq = seq[0:max_len]
        seq1 = seq.replace('B', 'N')  # asparagine N / aspartic acid  D - asx - B
        seq2 = seq1.replace('Z', 'Q')  # glutamine Q / glutamic acid  E - glx - Z
        seq3 = seq2.replace('U',
                            'C')  # selenocisteina, the closest is the cisteine. but it is a different aminoacid . take care.
        seq4 = seq3.replace('O', 'K')  # Pyrrolysine to lysine
        seq = seq4.replace('X', '')  # unknown character eliminated
        return seq

@timer
def variance_ngram(threshold = 0.1):
    print('--Getting NGRAMS Var ')
    data1 = pd.read_csv('counts_por_seq.csv')
    data1 = data1.drop(data1.columns[[0, 1]], axis=1)
    sel = VarianceThreshold(threshold)
    transf = sel.fit_transform(data1)
    # original dataset without columns
    ngram_selected = []
    column_selected = sel.get_support(indices=True)
    for i in column_selected:
        ngram_selected.append(data1.columns[i])
    return ngram_selected

def vector_seq(seq, ngram_selected=None , matrix_file:str = None):
    w2vmodel = wv(ngram_len=3 , emb_matrix_file= matrix_file)
    vector = w2vmodel.convert_seq2vec(method=1, sequence=seq , ngram_remove=ngram_selected )
    return vector

# try stuff

def get_index_of_ngrams_to_consider():
    # list of ngrams to remove/to keep
    # go to word2vec and get indices of this ngrams
    pass

    # per seq select n grams to keep
def vector_seq_without_sp_ngrams(vector_seq, n_grams_index):
    new_vector = vector_seq [n_grams_index]
    return new_vector


data = pd.read_csv('datasets/ecpred_uniprot_uniref_90 (2).csv')
data=data.dropna(subset=['sequence'])
print(data.shape)
data =data[~data['sequence'].str.contains("!")]
print(data.shape)
data=data.sample(frac=1)

get_ec_1_level(data)
data['ec_num'] = [x[0] for x in data['ec_number1']]
data = data.loc[data['ec_num']!=0]
data = data.loc[data['ec_num']!='0']

#data = data[:250]
print(data.shape)
y= data['ec_num']

seqs = data['sequence']
max_len = 500
#ngram_remove = variance_ngram()
seqs_new=map(lambda seq:transform_seq(seq, max_len),seqs)
seqs_new = list(seqs_new)
#seqs_vector = map(vector_seq,seqs_new)
#seqs_vector = map(lambda p:vector_seq(p, ngram_remove),seqs_new)



w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_sg_2gram_100dim_10ep_nneg.csv', ngram_len=2 , sequence_max_len= max_len , vectordim= 100)
#w2v = wv(emb_matrix_file='protVec_100d_3grams.csv', ngram_len=3 , sequence_max_len= max_len)
#w2v = wv(model_file='/home/igomes/Bumblebee/word2vec_30.model', ngram_len=3 , sequence_max_len= max_len)
#w2v = wv(emb_matrix_file='protVec_100d_3grams.csv', ngram_len=3 , sequence_max_len= max_len)

seqs_vector = w2v.convert_sequences2vec(method=1,sequences=seqs_new,padding=True, array_flat=False)
# seqs_vector = []
# for i in seqs_new:
#     #vector = w2v.convert_seq2vec(method=3, sequence=i, ngram_remove=ngram_selected)
#     vector = w2v.convert_seq2vec(method=3, sequence=i)
#     seqs_vector.append(vector)
# seqs_vector = np.array(seqs_vector)

print('seqs vectorized')
# # BINARIZE LABELS
# def binarize_labels(fps_y): # for single
#     test = pd.Series(fps_y)
#
#     # mlb = MultiLabelBinarizer()
#     # hot = mlb.fit_transform(test)
#     # res = pd.DataFrame(hot,
#     #                    columns=mlb.classes_,
#     #                    index=test.index)
#     fps_y = [item for sublist in fps_y for item in sublist] # this line is because they are retrieved as a list
#     encoder = LabelEncoder()
#     encoder.fit(fps_y)
#     encoded_Y = encoder.transform(fps_y)
#     classes = encoder.classes_
#     fps_y = np_utils.to_categorical(encoded_Y) # convert integers to dummy variables (i.e. one hot encoded)
#
#     from sklearn.preprocessing import OneHotEncoder
#
#
#     return encoded_Y, fps_y, classes

X = seqs_vector

counts = Counter(x for xs in y for x in set(xs))
counts.most_common()
df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
df_sorted = df.sort_values(by=[0], ascending=False)
print(df_sorted)
print('splitting')
df_x, df_x_val, y_, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
df_x_train, df_x_test, y_train, y_test = train_test_split(df_x, y_, test_size=0.10, random_state=42, stratify=y_)
print('train_x', df_x_train.shape)
print('test_x', df_x_test.shape)


# print(y_train)
# train_data = tf.data.Dataset.from_tensor_slices((np.array(df_x_train), np.array(y_train)))
# val_data = tf.data.Dataset.from_tensor_slices((np.array(df_x_val), np.array(y_val)))
# #vector_dim=100
# # seq_len = 502
# # x_train shape = (samples, seqlen, timestep aka vector_dim)
# batch_size = 32
# train_data = train_data.batch(batch_size)
# val_data = val_data.batch(batch_size)
#
# # Disable AutoShard.
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# train_data = train_data.with_options(options)
# val_data = val_data.with_options(options)
#
# print(train_data)

def lstm(input_dim, number_classes, vector_dim = 100 ,
                              optimizer='Adam',
                              lstm_layers=(64,),
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              dropout_rate=(0.1,),
                              l1=1e-5, l2=1e-4,
                              dense_layers=(64,),
                              dropout_rate_dense=(0.1,),
                              dense_activation="relu",loss='categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, vector_dim,), dtype='float32', name='main_input'))
        # add initial dropout
        model.add(Masking(mask_value=0))
        model.add(LSTM(units=64, return_sequences=False, activation=activation,
                       recurrent_activation=recurrent_activation,
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       dropout=0.1, recurrent_dropout=0.0))
        model.add(Dense(units=64, activation=dense_activation,
                        kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model

def bilstm_simple(input_dim, number_classes, vector_dim = 100,
                              optimizer='Adam',
                              lstm_layers=(256,),
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              dropout_rate=(0.3,),
                              l1=1e-5, l2=1e-4,
                              dense_layers=(128,64),
                              dropout_rate_dense=(0.1,0.1),
                              dense_activation="relu",loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, vector_dim,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers) - 1):
            model.add(
                Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0)                )
            )
            # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
            #          recurrent_activation=recurrent_activation,
            #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            #          dropout=dropout_rate[layer], recurrent_dropout=0.0)))

        # add last lstm layer
        model.add(
            Bidirectional(
            LSTM(units=lstm_layers[-1], return_sequences=False,
                 activation=activation, recurrent_activation=recurrent_activation,
                 kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                 dropout=dropout_rate[-1], recurrent_dropout=0.0))
                )
        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model

from tensorflow.keras import backend as K

class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()

#
#turn context


def bilstm_attention(input_dim, number_classes,
                     vector_dim=100,
                     optimizer='Adam',
                     lstm_layers=(256,),
                     activation='tanh',
                     recurrent_activation='sigmoid',
                     dropout_rate=(0.3,),
                     l1=1e-5, l2=1e-4,
                     dense_layers=(128,64),
                     dropout_rate_dense=(0.1,0.1),
                     dense_activation="relu", loss='sparse_categorical_crossentropy'):
    with strategy.scope():
        model = Sequential()
        # input dim timesteps = seq size , features. 21 features per character
        model.add(Input(shape=(input_dim, vector_dim,), dtype='float32', name='main_input'))
        # add initial dropout

        # model.add(Masking(mask_value=0, input_shape=(n_in, 1)))
        model.add(Masking(mask_value=0))
        for layer in range(len(lstm_layers)):
            model.add(Bidirectional(
                LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
                     recurrent_activation=recurrent_activation,
                     kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                     dropout=dropout_rate[layer], recurrent_dropout=0.0))
            )
            # model.add(LSTM(units=lstm_layers[layer], return_sequences=True, activation=activation,
            #          recurrent_activation=recurrent_activation,
            #          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            #          dropout=dropout_rate[layer], recurrent_dropout=0.0)

        # receives LSTM with return sequences =True

        # add attention
        #model.add(Attention(return_sequences=False)) # receive 3D and output 2D
        model.add(attention())
        # a, context = attention()(model)

        # add denses
        for layer in range(len(dense_layers)):
            model.add(Dense(units=dense_layers[layer], activation=dense_activation,
                            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate_dense[layer]))

        # Add Classification Dense, Compile model and make it ready for optimization
        model.add(Dense(number_classes, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        return model

dl=DeepML(x_train = df_x_train, y_train = y_train, x_test=df_x_test, y_test= y_test,
          number_classes=7, problem_type='multiclass',
          x_dval=df_x_val, y_dval=y_val,
          epochs=500, batch_size=512,
          path='', report_name='biLSTM_att_bumblebee_2g_100d_skip_w2v', verbose=1,early_stopping_patience=50)

print(df_x_train.shape)
vector_size = df_x_train.shape[1]
final_units = 7
model = KerasClassifier(build_fn=bilstm_attention, input_dim=vector_size, number_classes=final_units)
model = dl.run_model(model)

#
scores,report, cm, cm2 = dl.model_complete_evaluate(x_test=df_x_test,y_test=y_test)
# tf.Keras.clear_session()
tf.keras.backend.clear_session()

print('scores', scores)



