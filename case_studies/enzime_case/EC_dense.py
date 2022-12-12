import pickle
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from time import sleep
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
#strategy = tf.distribute.MirroredStrategy()
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from propythia.shallow_ml import ShallowML
from collections import Counter
from word_embedding_main import WordEmbedding as wv
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
# todo
# go check the notebooks for examples at https://github.com/BioSystemsUM/propythia
# examples and docs (pdf and quickstart)
#############################################################################################################
# 1 define X and Y and split into Test and Train
# train_test_split scikit learn

import time
import functools
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
#data = data[:2500]


# for seq in seqs_vector2:
#     df_x_train = seq
#     print(df_x_train.shape)
#     break

# get_seqs_vectors(data[0:25])
# data.to_csv('datasets/ecpred_uniprot_uniref_90_w_vectores_method_2_test.csv')
# print('Dataset Saved..')
# print('seqs vectorized')
# data = pd.read_csv('datasets/ecpred_uniprot_uniref_90_w_vectores_method_2.csv')
get_ec_1_level(data)
# colm_exclude = ['Unnamed: 0', 'uniref_90', 'Entry', 'Protein names', 'ec_number','sequence', 'pfam', 'supfam', 'go', 'ec_number1', 'ec_single_label']
# X = data.loc[:, ~data.columns.isin(colm_exclude)]
#df.drop(df.index[df['line_race'] == [0]], inplace = True)
data['ec_num'] = [x[0] for x in data['ec_number1']]

data = data.loc[data['ec_num']!=0]
data = data.loc[data['ec_num']!='0']

#data = data[:240]
print(data.shape)
y= data['ec_num']

seqs = data['sequence']
max_len = 500
#ngram_remove = variance_ngram()
seqs_new=map(lambda seq:transform_seq(seq, max_len),seqs)
seqs_new = list(seqs_new)
#seqs_vector = map(vector_seq,seqs_new)
#seqs_vector = map(lambda p:vector_seq(p, ngram_remove),seqs_new)


w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_skip_3gram_1dim_10ep_nneg.csv', ngram_len=3 , sequence_max_len= max_len)
#w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/data_processing/w2v_cbow_3gram_100dim.csv', ngram_len=3 , sequence_max_len= max_len)
#w2v = wv(emb_matrix_file='protVec_100d_3grams.csv', ngram_len=3 , sequence_max_len= max_len)
#w2v = wv(model_file='/home/igomes/Bumblebee/word2vec_30.model', ngram_len=3 , sequence_max_len= max_len)

print('star vec ')
#seqs_vector = w2v.convert_sequences2vec(method=1,sequences=seqs_new,padding=True, array_flat=True)
seqs_vector = []
for i in seqs_new:
    #vector = w2v.convert_seq2vec(method=3, sequence=i, ngram_remove=ngram_selected)
    vector = w2v.convert_seq2vec(method=3, sequence=i)
    seqs_vector.append(vector)
seqs_vector = np.array(seqs_vector)

print('seqs vectorized')
X = seqs_vector

counts = Counter(x for xs in y for x in set(xs))
counts.most_common()
df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
df_sorted = df.sort_values(by=[0], ascending=False)
print(df_sorted)

print('splitting')
#df_x_train, df_x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

df_x, df_x_val, y_, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
df_x_train, df_x_test, y_train, y_test = train_test_split(df_x, y_, test_size=0.10, random_state=42, stratify=y_)
print('train_x', df_x_train.shape)
print('test_x', df_x_test.shape)

# vector_dim=100
# seq_len = 502
# x_train shape = (samples, seqlen, timestep aka vector_dim)

# print('aquiiiiiiiiiiiiiiiiiiiiiii')
# print(df_x_train.shape[1], df_x_train.shape[2])
from propythia.deep_ml import DeepML

dl=DeepML(x_train = df_x_train, y_train = y_train, x_test=df_x_test, y_test= y_test,
          number_classes=7, problem_type='multiclass',
          x_dval=df_x_val, y_dval=y_val,
          epochs=500, batch_size=1024,
          path='', report_name='dnn_mth3_bumble_10ep', verbose=1, early_stopping_patience=50)

dl.run_dnn_simple(input_dim=df_x_train.shape[1],
                       optimizer='Adam',
                       hidden_layers=(1048,1048,1048),
                       dropout_rate=(0.3,),
                       batchnormalization=(True,),
                       l1=1e-5, l2=1e-4,
                       final_dropout_value=0.1,
                       initial_dropout_value=0.0,
                       loss_fun='sparse_categorical_crossentropy', activation_fun='softmax',
                       cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=40,
                       scoring=make_scorer(matthews_corrcoef))

# dl.run_lstm_simple(input_dim=df_x_train.shape[0],
#                         optimizer='Adam',
#                         bilstm=False,
#                         lstm_layers=(128, 64),
#                         dense_layers=(64,),
#                         activation='tanh',
#                         recurrent_activation='sigmoid',
#                         dense_activation="relu",
#                         l1=1e-5, l2=1e-4,
#                         dropout_rate=(0.3,), recurrent_dropout_rate=(0.3,),
#                         dropout_rate_dense=(0.3,),
#                         batchnormalization = (True,),
#                         loss_fun = None, activation_fun = None,
#                         cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
#                         scoring=make_scorer(matthews_corrcoef))
# dl.run_cnn_1D(input_dim=(df_x_train.shape[1]),
#                    optimizer='Adam',
#                    filter_count=(32, ),  # define number layers
#                    padding='same',
#                    strides=1,
#                    kernel_size=(3,),  # list of kernel sizes per layer. if number will be the same in all numbers
#                    cnn_activation='relu',
#                    kernel_initializer='glorot_uniform',
#                    dropout_cnn=(0.1, ),
#                    # list of dropout per cnn layer. if number will be the same in all numbers
#                    max_pooling=(True,),
#                    pool_size=(2,), strides_pool=1,
#                    data_format_pool='channels_first',
#                    dense_layers=(64, ),
#                    dense_activation="relu",
#                    dropout_rate=(0.3,),
#                    l1=1e-5, l2=1e-4,
#                    loss_fun = None, activation_fun = None,
#                    cv=None, optType=None, param_grid=None, n_iter_search=15, n_jobs=1,
#                    scoring=make_scorer(matthews_corrcoef))

scores,report, cm, cm2 = dl.model_complete_evaluate(x_test=df_x_test,y_test=y_test)
# tf.Keras.clear_session()
tf.keras.backend.clear_session()
print('scores', scores)



# 2 define class ML from ProPythia and run something

# # file names
# report = 'name_of_Experiment'
# report_ml = str(report + '.txt')
# path_for_roc = str(report + '_roc.png')
# path_for_val_curve = str(report + '_valcurve.png')
# path_for_learn_curve = str(report)
# path_fi = str(report + '_fi.png')