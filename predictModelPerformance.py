# -*- encoding:utf-8 -*-
"""
Author: RediscoverM
Email: 18240439947@163.com
Name: pre.py
Time: 2019/3/13  9:02
Others：

"""

import urllib.request
import numpy
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual):
        return -1
    tp = 0.0; fp = 0.0; fn = 0.0; tn = 0.0
    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0
            else:
                fn += 1.0
        else:
            if predicted[i] < threshold:
                tn += 1.0
            else:
                fp += 1.0
    return [tp, fn, fp, tn]

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
data = urllib.request.urlopen(target_url)
xlist = []
labels = []
for line in data:
    row = line.strip().decode().split(",")
    if row[-1] == 'M':
        labels.append(1.0)
    else:
        labels.append(0.0)
    row.pop()
    floatRow = [float(num) for num in row]
    xlist.append(floatRow)

indices =range(len(xlist))
xListTest = [xlist[i] for i in indices if i % 3 ==0]
xListTrian = [xlist[i] for i in indices if i % 3 !=0]
labelsTest = [labels[i] for i in indices if i % 3 ==0]
labelsTrain = [labels[i] for i in indices if i % 3 !=0]

xTrain = numpy.array(xListTrian)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)


print("xTrain array", xTrain.shape)
print("yTrain array", yTrain.shape)
print("xTest array", xTest.shape)
print("yTest array", yTest.shape)


rocksVMinesModel = linear_model.LinearRegression()
rocksVMinesModel.fit(xTrain, yTrain)

trainingPredictions = rocksVMinesModel.predict(xTrain)
print("Some values predicted by model", trainingPredictions[0: 5], trainingPredictions[-6:-1])

confusionMatTrain = confusionMatrix(trainingPredictions, yTrain, 0.5)
tp = confusionMatTrain[0]
fn = confusionMatTrain[1]
fp = confusionMatTrain[2]
tn = confusionMatTrain[3]

print("tp = " + str(tp))
print("fn = " + str(fn))
print("fp = " + str(fp))
print("tn = " + str(tn))
print('\n\n\n')
print("---------------------------------------------------------")
print('\n\n\n')

testPredictions = rocksVMinesModel.predict(xTest)
conMatTest = confusionMatrix(testPredictions, yTest, 0.5)
tp = confusionMatTrain[0]
fn = confusionMatTrain[1]
fp = confusionMatTrain[2]
tn = confusionMatTrain[3]

print("tp = " + str(tp))
print("fn = " + str(fn))
print("fp = " + str(fp))
print("tn = " + str(tn))
print('\n\n\n')
print("*********************************************************")
print('\n\n\n')

fpr, tpr, thresholds = roc_curve(yTrain, trainingPredictions)
roc_auc = auc(fpr, tpr)
print("Auc for in-sample ROC curve : %f" % roc_auc)

pl.figure()
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.title("ROC")
pl.legend(loc = "lower right")

fpr, tpr, thresholds = roc_curve(yTest, testPredictions)
roc_auc = auc(fpr, tpr)
print("Auc for in-sample ROC curve : %f" % roc_auc)

pl.figure()
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f' % roc_auc)
pl.plot([0, 1], [0, 1], 'k-')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.title("ROC")
pl.legend(loc = "lower right")
pl.show()


