import numpy as np
from Bio import SeqIO
from nltk import bigrams
import numpy
import re

from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import np_utils
from word_embbeding1 import WordEmbedding
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef

texts = []
for index, record in enumerate(SeqIO.parse('/home/igomes/Bumblebee/ubisites_caso/datasets/testset1.fasta', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1]
        #temp_str = temp_str + " " +item[0]
    texts.append(temp_str)
#
seq=[]
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
    doc = re.sub(stop, '', doc)
    seq.append(doc.split())
# #
# w2v_model = Word2Vec.load('word2vec.model')
# embedding_matrix = w2v_model.wv.vectors
# #
# vocab_list = list(w2v_model.wv.vocab.keys())

#
w2v_model = WordEmbedding(ngram_len=2, emb_matrix_file='trained_word_embeddings/word2vecmodel_skipgram_matrix.csv')
embedding = w2v_model.get_emb_matrix()
vocab_list = list(embedding.keys())
embedding_matrix = []
for i in embedding.values():
    embedding_matrix.append(list(i))
embedding_matrix = numpy.array([numpy.array(xi) for xi in embedding_matrix])


#
word_index = {word: index for index, word in enumerate(vocab_list)}
#
def get_index(sentence):
    global word_index
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence
#

X_data_test= np.array(list(map(get_index, seq)))
Y_data_test = np.load("/home/igomes/Bumblebee/ubisites_caso/datasets/y_testset.npy")
#
model = load_model('model_bumblebee_es.h5')
#model = load_model('model_bumblebee.h5')

#preds = model.predict_classes(X_data)

scores = {}

try:
    y_prob = model.predict_proba(X_data_test)
    y_pred = model.predict(X_data_test)

except:
    y_prob = model.predict(X_data_test)
    y_pred= np.argmax(y_prob,axis=1)



scores['Accuracy'] = accuracy_score(Y_data_test, y_pred)
scores['MCC'] = matthews_corrcoef(Y_data_test, y_pred)
scores['log_loss'] = log_loss(Y_data_test, y_prob)


# binary
scores['f1 score'] = f1_score(Y_data_test, y_pred)
scores['roc_auc'] = roc_auc_score(Y_data_test, y_pred)
precision, recall, thresholds = precision_recall_curve(Y_data_test, y_pred)
scores['Precision'] = precision
scores['Recall'] = recall
tn, fp, fn, tp = confusion_matrix(Y_data_test, y_pred).ravel()
# scores['fdr'] = float(fp) / (tp + fp)
scores['sn'] = float(tp) / (tp + fn)
scores['sp'] = float(tn) / (tn + fp)


acc = accuracy_score(Y_data_test, y_pred)
mcc = matthews_corrcoef(Y_data_test, y_pred)
print('accuracy', acc)
print('MCC', mcc)

report = classification_report(Y_data_test, y_pred, output_dict=False)
cm = confusion_matrix(Y_data_test, y_pred)
print(cm)
print(report)