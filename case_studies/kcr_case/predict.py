

reportname = ''

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