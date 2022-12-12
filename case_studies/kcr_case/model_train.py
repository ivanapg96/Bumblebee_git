import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, LSTM, Dropout, Bidirectional, Reshape , GRU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from sklearn import metrics
import random
from tensorflow.keras.regularizers import l1, l2
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
reportname = 'kcr_gru_metrics'
seed = 52
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.debugging.set_log_device_placement(True)
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

from tensorflow.keras.layers import *
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

#Using TensorFlow backend.
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(test_Y, pre_test_y):
    #calculate the F1-score
    Precision = precision(test_Y, pre_test_y)
    Recall = recall(test_Y, pre_test_y)
    f1 = 2 * ((Precision * Recall) / (Precision + Recall + K.epsilon()))
    return f1

def TP(test_Y,pre_test_y):
    #calculate numbers of true positive samples
    TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
    return TP

def FN(test_Y,pre_test_y):
     #calculate numbers of false negative samples
    TP = K.sum(K.round(K.clip(test_Y * pre_test_y, 0, 1)))#TP
    P=K.sum(K.round(K.clip(test_Y, 0, 1)))
    FN = P-TP #FN=P-TP
    return FN

def TN(test_Y,pre_test_y):
    #calculate numbers of True negative samples
    TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
    return TN

def FP(test_Y,pre_test_y):
    #calculate numbers of False positive samples
    N = (-1)*K.sum(K.round(K.clip(test_Y-K.ones_like(test_Y), -1, 0)))#N
    TN=K.sum(K.round(K.clip((test_Y-K.ones_like(test_Y))*(pre_test_y-K.ones_like(pre_test_y)), 0, 1)))#TN
    FP=N-TN
    return FP

def dnn_model(train_X, train_Y, test_X, test_Y, lr, epoch, batch_size):
    train_X = np.expand_dims(train_X, 2)
    test_X = np.expand_dims(test_X, 2)
    inputs = Input(shape = (train_X.shape[1], train_X.shape[2]))
    x = Conv1D(128, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu')(inputs)
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation = 'relu',kernel_regularizer = l2(1e-5))(x)
    x = Dense(16, activation = 'relu',kernel_regularizer = l2(1e-5))(x)
    x = Dense(8, activation = 'relu',kernel_regularizer = l2(1e-5))(x)
    predictions = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = predictions)
    print("model")
    model.compile(optimizer = 'RMSProp',
                  loss = 'mean_squared_error',
                  metrics = ['acc',precision,recall,f1,TP,FN,TN,FP])
    print("compile")
    model.fit(train_X, train_Y, epochs = epoch, batch_size = 32, validation_data = (test_X, test_Y), shuffle = True)
    model.save('6fea_idx12Scale_infoGainSorted50.h5') #save model
    pre_test_y = model.predict(test_X, batch_size = 50)
    pre_train_y = model.predict(train_X, batch_size = 50)
    test_auc = metrics.roc_auc_score(test_Y, pre_test_y)
    train_auc = metrics.roc_auc_score(train_Y, pre_train_y)
    print("train_auc: ", train_auc)
    print("test_auc: ", test_auc)
    return test_auc

def dnn_model_2(train_X, train_Y, test_X, test_Y, lr, epoch, batch_size):
    train_X = np.expand_dims(train_X, 2)
    test_X = np.expand_dims(test_X, 2)
    inputs = Input(shape = (train_X.shape[1], train_X.shape[2]))
    x = Conv1D(128, kernel_size = 8, strides = 1, padding = 'valid', activation = 'relu')(inputs)
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, kernel_size = 8, strides = 1, padding = 'valid', activation = 'relu')(x)
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, kernel_size = 8, strides = 1, padding = 'valid', activation = 'relu')(x)
    x = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu',kernel_regularizer = l2(1e-5))(x)
    predictions = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = predictions)
    print("model")
    #optimizer RMSProp

    model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy',
                  metrics = ['acc',precision,recall,f1,TP,FN,TN,FP])
    print("compile")
    model.fit(train_X,
              train_Y,
              epochs = epoch,
              batch_size = batch_size,
              validation_data = (test_X, test_Y),
              shuffle = True,
              callbacks=[es])
    model.save(reportname +'.h5') #save model
    scores_crossval = {}
    y_pred = model.predict(test_X)
    pre_train_y = model.predict(train_X)
    pre_test_y = []
    for i in y_pred:
        list2 = []
        for x in i:
            list2.append(int(round(x, 0)))
        pre_test_y.append(list2)
    test_auc = metrics.roc_auc_score(test_Y, pre_test_y)
    train_auc = metrics.roc_auc_score(train_Y, pre_train_y)
    scores_crossval['Accuracy'] = accuracy_score(test_Y, pre_test_y)
    scores_crossval['MCC' ] = matthews_corrcoef(test_Y, pre_test_y)
    # binary
    # scores_crossval['f1 score'+ str(train_fold)] = f1_score(Y[train_fold], Y[validate_fold])
    scores_crossval['roc_auc'] = roc_auc_score(test_Y, pre_test_y)
    #precision_, recall_, thresholds = precision_recall_curve(test_Y, pre_test_y)
    tn, fp, fn, tp = confusion_matrix(test_Y, pre_test_y).ravel()
    # scores['fdr'] = float(fp) / (tp + fp)
    scores_crossval['sn'] = float(tp) / (tp + fn)
    scores_crossval['sp'] = float(tn) / (tn + fp)
    print("train_auc: ", train_auc)
    print("test_auc: ", test_auc)
    return test_auc , scores_crossval


def dnn_model_3(train_X, train_Y, test_X, test_Y, lr, epoch, batch_size):
    print(train_X.shape)
    train_X = train_X.reshape(train_X.shape[0],31,5)
    test_X = test_X.reshape(test_X.shape[0], 31, 5)
    #tamnho de sequenceis = 31 | dimensao do vector = 5 | ngram = 1
    inputs = Input(shape = (train_X.shape[1], train_X.shape[2]))
    #x = Bidirectional(LSTM(128, activation='relu'))(inputs)
    x = GRU(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = GRU(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(1e-5))(x)
    predictions = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = inputs, outputs = predictions)
    print("model")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc', precision, recall, f1, TP, FN, TN, FP])
    #optimizer RMSProp
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=[tf.keras.metrics.CategoricalAccuracy()])
    print("compile")
    model.fit(train_X, train_Y, epochs = epoch, batch_size = batch_size, validation_data = (test_X, test_Y), shuffle = True,callbacks=[es,reduce_lr])
    model.save(reportname +'.h5') #save model
    scores_crossval = {}
    y_pred = model.predict(test_X)
    pre_train_y = model.predict(train_X)
    pre_test_y = []
    for i in y_pred:
        list2 = []
        for x in i:
            list2.append(int(round(x, 0)))
        pre_test_y.append(list2)
    test_auc = metrics.roc_auc_score(test_Y, pre_test_y)
    train_auc = metrics.roc_auc_score(train_Y, pre_train_y)
    scores_crossval['Accuracy'] = accuracy_score(test_Y, pre_test_y)
    scores_crossval['MCC'] = matthews_corrcoef(test_Y, pre_test_y)
    # binary
    # scores_crossval['f1 score'+ str(train_fold)] = f1_score(Y[train_fold], Y[validate_fold])
    scores_crossval['roc_auc'] = roc_auc_score(test_Y, pre_test_y)
    precision_, recall_, thresholds = precision_recall_curve(test_Y, pre_test_y)
    tn, fp, fn, tp = confusion_matrix(test_Y, pre_test_y).ravel()
    # scores['fdr'] = float(fp) / (tp + fp)
    scores_crossval['sn'] = float(tp) / (tp + fn)
    scores_crossval['sp'] = float(tn) / (tn + fp)
    print("train_auc: ", train_auc)
    print("test_auc: ", test_auc)
    return test_auc, scores_crossval

#TRAIN
sequences_pos = np.array(pd.read_csv('/home/igomes/Bumblebee/deep_kcr_caso/data/train/pos_train.txt'))
sequences_neg =  np.array(pd.read_csv('/home/igomes/Bumblebee/deep_kcr_caso/data/train/neg_train.txt'))
sequences = np.append(sequences_pos,sequences_neg)



from word_embedding_main import WordEmbedding
w2v_model = WordEmbedding(ngram_len=1,emb_matrix_file='/home/igomes/Bumblebee/deep_kcr_caso/trained_models/w2v_sg_20_5.csv')
data = w2v_model.convert_sequences2vec(method=1,sequences=sequences ,array_flat=True)

print(data.shape)
# split data and output result
#data = np.array(pd.read_csv("6fea_idx12Scale_infoGainSorted50.csv"))#inputfile
# X1 = data[0:6975, 1:]#6975 is the number of positive samples in training set, '1' is the label of positive sample
# Y1 = data[0:6975, 0]#'0' is the label of negative sample
# X2 = data[6975:, 1:]
# Y2 = data[6975:, 0]
X1 = data[0:6974 ]
Y1 = [1] * 6974

X2 = data[6974:]
Y2 = [0] * 6974
print ( 'x1.shape:', X1.shape )
print ( 'x2.shape:',X2.shape)
X = np.concatenate([X1, X2], 0)
Y = np.concatenate([Y1, Y2], 0)
#Y = Y.reshape((Y.shape[0], -1))
#print (X)
print ("X.shape:", X.shape)
print ("Y.shape:", Y.shape)
# simple early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
patience_es = 50
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                   patience=patience_es)
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
lr = 0.001 #learning rate 0.2
epoch = 500 #500
batch_size = 128#32
#kf = KFold(n_splits = 10, shuffle = True, random_state = seed)
kf = KFold(n_splits = 5, shuffle = True, random_state=seed)
kf = kf.split(X)
import pandas as pd
scores_vales = pd.DataFrame([])
test_aucs = []
for i, (train_fold, validate_fold) in enumerate(kf):
    print("\n\ni: ", i)
    test_auc , scores = dnn_model_3(X[train_fold], Y[train_fold], X[validate_fold], Y[validate_fold], lr, epoch, batch_size)
    print(test_auc)
    print(scores)
    test_aucs.append(test_auc)
    scores_vales = scores_vales.append(scores, ignore_index=True)



#TEST
sequences_pos = np.array(pd.read_csv('/home/igomes/Bumblebee/deep_kcr_caso/data/test/pos_test.txt'))
sequences_neg =  np.array(pd.read_csv('/home/igomes/Bumblebee/deep_kcr_caso/data/test/neg_test.txt'))
sequences = np.append(sequences_pos,sequences_neg)
print(sequences_pos.shape)
print(sequences_neg.shape)
data = w2v_model.convert_sequences2vec(method=1,sequences=sequences ,array_flat=True)
print('datashape' , data.shape)
X1 = data[0:2988 ]
Y1 = [1] * 2988
X2 = data[2988:]
Y2 = [0] * 2988
print ( 'x1.shape:', X1.shape )
print ( 'x2.shape:',X2.shape)
X_test = np.concatenate([X1, X2], 0)
Y_test = np.concatenate([Y1, Y2], 0)
X_test = X_test.reshape(X_test.shape[0],31,5) #comentar se nao for LSTM


from tensorflow.keras.models import load_model
model_name = reportname + '.h5'
model = load_model(model_name,
                         custom_objects={'precision': precision,'recall':recall,'f1':f1,'TP':TP,'FN':FN,'TN':TN,'FP':FP})

#######################################################
# # model = load_model('pcsf.h5')
# accuracy = model_back.evaluate(X,Y)
# # print 'loss', loss
# print('accuracy', accuracy)
# maxprobability = model_back.predict(X)
# np.set_printoptions(threshold=np.inf)
# print(maxprobability)
# fw = open('Result.txt','w') #define result outputFile
# myprob = "\n".join(map(str, maxprobability[:, 0]))
# fw.write(myprob)
# predictclass = model_back.predict(X)
# predictclass = np.argmax(predictclass,axis=1)
# print(predictclass)
# y_pred = predictclass
# Y_data_test = Y

#####################

y_prob = model.predict(X_test, batch_size = 50)
y_pred = []
for i in y_prob:
    list2 = []
    for x in i:
        list2.append(int(round(x, 0)))
    y_pred.append(list2)

test_auc = metrics.roc_auc_score(Y_test, y_prob)
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

#Print Result Report

means = scores_vales.mean(axis=0)
w = open( reportname+"_results.txt", "a")
model.summary(print_fn=lambda x: w.write(x + '\n'))
w.writelines( 'seed: ' + str(seed))
w.write('\n')
w.writelines('epochs: ' + str(epoch))
w.write('\n')
w.writelines( 'batch_size: ' + str(batch_size))
w.write('\n')

if es:
    w.writelines('EarlyStopping patience: ' + str(patience_es))
    w.write('\n')
    #
    # w.writelines('Reduce Learning Rate patience: ' + str(patience_lr))
    # w.write('\n')
    # w.writelines('rlr min: '+ str(reduce_lr_min))
    # w.write('\n')
    # w.writelines('rlr factor: ' + str(reduce_lr_factor))
    # w.write('\n')

w.write('Metrics for CrossVal')
w.write('\n')
fAsString = scores_vales.to_string(header=True, index=False)
w.write(fAsString)
w.write('\n')
w.write('Means of Crossvalues Scores: ')
w.write(means.to_string(header=True, index=False))
w.write('\n')


w.write('AUC for CrossVal')
w.write('\n')

for j in test_aucs:
    w.write(str(j) + ',')
    w.write('\n')
w.write('AUC Mean for CrossVal')
w.write(str(np.mean(test_aucs)) + '\n')

w.write('Metrics for Independent Test')
w.write('\n')
for j , i in scores.items():
    w.write(str(j) + ',')
    w.write(str(i) + ',')
    w.write('\n')

w.write('confusion matrix:')
w.write('\n')
w.write(str(cm))
w.close()

#verificar as metrics
#tentar melhorar hyperparametros
#encontrar outro paper