import numpy as np
import sys
import h5py
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

filename=sys.argv[1]
instance=sys.argv[2]
myFile=h5py.File(filename,'r')
y_predict=myFile['predicts'][:]
y_test= np.array(myFile['labels'][:]) 
y_score= np.exp(np.array(myFile['scores'][:]))

accuracyScore=metrics.accuracy_score(y_test, y_predict, normalize=False)
f1Score=metrics.f1_score(y_test, y_predict, average='weighted')
precisionScore=metrics.precision_score(y_test, y_predict, average='weighted')
recallScore = metrics.recall_score(y_test, y_predict, average='weighted')
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=7)
aucScore = metrics.auc(fpr, tpr)

print datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print "AUCscore: ", aucScore
print "F-measure: ", f1Score
print "Recall: ", recallScore
print "Precision:", precisionScore
print "Accuracy: ", accuracyScore


with open('LRCL_result.txt','a') as f:
    f.write('INSTANCE: '+bytes(instance)+'\r\n--Date: '+bytes(datetime.datetime.now().strftime("%Y-%m-%d"))+'\r\n--f1_score : '+bytes(f1Score)+' --aucScore : '+bytes(aucScore) +' --recallScore : '+bytes(recallScore) +'--precision : '+bytes(precisionScore)+'--accuracy : '+bytes(accuracyScore)+'\r\n')

confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
print "confusion_matrix: ",confusion_matrix
np.save('confusion_matrix_'+bytes(instance),confusion_matrix)

classification_report = metrics.classification_report(y_test, y_predict)  
print "classification_report : ",classification_report
np.save('classification_report_'+bytes(instance),classification_report)

plt.hist(y_predict)
plt.xlim(0, 10)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.savefig(instance)



