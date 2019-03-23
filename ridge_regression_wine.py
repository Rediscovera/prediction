import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
import urllib.request

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = urllib.request.urlopen(target_url)
xlist = []
labels = []
names = []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().decode().split(";")
        firstLine = False
    else:
        row = line.strip().decode().split(";")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xlist.append(floatRow)

indices = range(len(xlist)) #样本个数
xListTest = [xlist[i] for i in indices if i%3 == 0]
xListTrain = [xlist[i] for i in indices if i%3 != 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain)
xTest = numpy.array(xListTest)
# 标签是固定的  不会改变
yTrain = numpy.array(labelsTrain)
yTest = numpy.array(labelsTest)
alphaList = [0.1**i for i in range(0, 7)]

rmsError=[]
for alph in alphaList:
    wineRidgeModel = linear_model.Ridge(alpha=alph)
    wineRidgeModel.fit(xTrain, yTrain)
    rmsError.append(numpy.linalg.norm((yTest-wineRidgeModel.predict(xTest)),2)/sqrt(len(yTest)))

for i in range(len(rmsError)):
    print(rmsError[i], alphaList[i])

x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.figure()

indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
wineRidgeModel = linear_model.Ridge(alpha=alph)
wineRidgeModel.fit(xTrain, yTrain)
errorVector = yTest - wineRidgeModel.predict(xTest)
plt.hist(errorVector)
plt.figure()

plt.scatter(wineRidgeModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.show()