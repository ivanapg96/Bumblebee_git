import pickle
import random
import pandas as pd
import numpy as np
import os
import time
import functools

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
    vector = w2vmodel.convert_seq2vec(method=3, sequence=seq , ngram_remove=ngram_selected )
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
print(len(y))

seqs = data['sequence']
max_len = 500
#ngram_remove = variance_ngram()
seqs_new=map(lambda seq:transform_seq(seq, max_len),seqs)
seqs_new = list(seqs_new)
print(len(seqs_new[0]))
#seqs_vector = map(vector_seq,seqs_new)
#seqs_vector = map(lambda p:vector_seq(p, ngram_remove),seqs_new)

w2v = wv(emb_matrix_file='/home/igomes/Bumblebee/protVec_100d_3grams.csv',
         ngram_len=3 , sequence_max_len= max_len , vectordim=100)
#seqs_vector = w2v.convert_sequences2vec(method=1,sequences=seqs_new,padding=True, array_flat=True)
seqs_vector = []
for i in seqs_new:
    #vector = w2v.convert_seq2vec(method=3, sequence=i, ngram_remove=ngram_selected)
    vector = w2v.convert_seq2vec(method=3, sequence=i)
    seqs_vector.append(vector)
seqs_vector = np.array(seqs_vector)

print('seqs vectorized')
X = seqs_vector
print(X.shape)
counts = Counter(x for xs in y for x in set(xs))
counts.most_common()
df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
df_sorted = df.sort_values(by=[0], ascending=False)
print(df_sorted)

# y2 = []
# for i in y:
#     y2.append(i[0])
print('splitting')
df_x_train, df_x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
print('train_x', df_x_train.shape)
print('test_x', df_x_test.shape)

# vector_dim=100
# seq_len = 502
# x_train shape = (samples, seqlen, timestep aka vector_dim)

# define ml class
# create Machine learning object

report = 'ml_svm_mth3_3ngram_100dim_protvec'
ml = ShallowML(x_train=df_x_train, x_test=df_x_test, y_train=y_train, y_test=y_test,
               report_name=report)
               #,columns_names=df_x_train.columns)  # o teu columns se calhar é None. depende se tiveres column names ou nao

# no caso svm
param_grid = {'clf__C': [0.1, 1.0, 10],
                        'clf__kernel': ['rbf'],
                        'clf__gamma': [0.001,0.0001]}
#               {'clf__C': [0.01, 0.1, 1.0, 10],
#                'clf__kernel': ['rbf'],
#                'clf__gamma': ['scale', 0.001, 0.0001]}]
# param_grid = {'clf__C': [0.1], 'clf__kernel': ['rbf']}
#param_grid = {'clf__C': [0.01,0.1,1.0 , 10]}
#['Model with rank: 1\n', 'Mean validation score: 0.653 (std: 0.003)\n', "Parameters: {'clf__leaf_size': 15, 'clf__n_neighbors': 2, 'clf__weights': 'distance'}\n", '\n']
#param_grid = {'clf__leaf_size': [15], 'clf__n_neighbors': [2], 'clf__weights': ['distance']}
#param_grid = {'clf__n_estimators': [100, 500],'clf__max_depth': [5, 10]}
# param_grid = {'clf__n_estimators': [100, 500],'clf__max_features': ['sqrt']}
#no caso kNN
# TRAIN BEST MODEL


# best_model = ml.train_best_model(model_name= None , model = 'svm', scaler=None,
#                                      score=make_scorer(matthews_corrcoef),
#                                      cv=3, optType='gridSearch',
#                                      param_grid=param_grid,
#                                      # podes ver as param grids em e podes subsitituir por uma mais pequena propythia/src/propythia/adjuv_functions/ml_deep/parameters_shallow.py
#                                      n_jobs=40, random_state=1, n_iter=15, refit=True)

##########################################
# scores = ml.cross_val_score_model(model_name = None,model='svm',
#                               score='accuracy',
#                               cv=3,
#                               n_jobs=10,
#                               random_state=1)

########################################{'clf__C': 10, 'clf__gamma': 0.0001, 'clf__kernel': 'rbf'}
from sklearn import svm
clf = svm.SVC(C=10,gamma=0.001,kernel='rbf')
# from sklearn import ensemble
# clf = ensemble.RandomForestClassifier(n_estimators=500,max_features='sqrt',random_state=1)
# from sklearn.neighbors import KNeighborsClassifier
# #Parameters:	{'clf__leaf_size': 15, 'clf__n_neighbors': 2, 'clf__weights': 'distance'}
# clf = KNeighborsClassifier(n_neighbors=2,weights= 'distance' ,leaf_size=15)
#
best_model = clf.fit(df_x_train, y_train)
y_pred = clf.predict(df_x_test)
score = matthews_corrcoef(y_test, y_pred)
#outros modelos a testar
#'svm', 'linear_svm', 'knn', 'sgd', 'lr','rf', 'gnb', 'nn','gboosting'

#se quiseres fazer so CV com um modelo ( e n procurar o melhor modelo )
# cross_val_score_model(model_name,model,
#                       score='accuracy',
#                       cv=10)

#Se quiseres salvar o modelo
filename = str(report + '.sav')
pickle.dump(best_model, open(filename, 'wb'))

#score test set
scores, report, cm, cm2 = ml.score_testset(classifier=best_model)
# scores, report, cm, cm2 = ml.score_testset()
print(scores)
print(report)
print(cm)



# curva ROC___
#
# try:
#     ml.plot_roc_curve(title='ROC curve for ' + str(best_model),
#                       path_save=path_for_roc, show=False)
# except Exception as e:
#     print(str(e))
# #
# # fazer plot de uma learning curve. ver os scores e cv scores consoante o numero de exemplos que dás ao modelo
# try:
#     ml.plot_learning_curve(n_jobs=10, train_sizes=np.linspace(.1, 1.0, 5),
#                            path_save=path_for_learn_curve, show=True, scalability=True,
#                            performance=True)
# except Exception as e:
#     print(str(e))


# validation curve___


# try:
#     ml.plot_validation_curve(param_name='algumnomedeparam', param_range=['uma lista com valores para o parametro'],
#                       classifier=None,
#                       cv=5,
#                       score=None, title="Validation Curve",
#                       xlab="parameter range", ylab="MCC", n_jobs=1, show=False,
#                       path_save='plot_validation_curve')
# except Exception as e:
#     print(str(e))



#Feature importance___

# try:
#         ml.features_importances_df(top_features=30, model_name=best_model)
# except Exception as e:
#     print(str(e))
# try:
#     ml.features_importances_plot(top_features=20, show=False, model_name=best_model,
#                                  path_save=path_fi, column_to_plot=0,
#                                  kind='barh', figsize=(9, 7), color='r', edgecolor='black')
# except Exception as e:
#     print(str(e))