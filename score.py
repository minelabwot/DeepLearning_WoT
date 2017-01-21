import h5py

myFile=h5py.File('predictValue.h5','r')
y_predict=myFile['predicts']
y_test=myFile['labels']

from sklearn import metrics
accuracyScore=metrics.accuracy_score(y_test, y_predict, normalize=False)
f1Score=metrics.f1_score(y_test, y_predict, average='weighted')
precisionScore=metrics.precision_score(y_test, y_predict, average='weighted')
print "Precision:", precisionScore

print "Recall: ", metrics.recall_score(y_test, y_predict, average='weighted')

print "F-measure: ", f1Score

print "Accuracy: ", accuracyScore

with open('torch_end_scores.txt','a') as f:
    f.write('-f1_score is :'+bytes(f1Score)+'the precision is '+bytes(precisionScore)+'--the accuracy is:'+bytes(accuracyScore)+'\r\n')





