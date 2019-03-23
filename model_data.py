import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
import urllib.request

def xattr_select(x, idxSet):
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return xOut

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

attributeList = []
index = range(len(names)-1) #去掉标签列剩下的列总数
# print(index)
indexSet = set(index)
# print(indexSet)
indexSeq = []
oosError = []

for i in index:
    attSet = set(attributeList)
    attTrySet = indexSet - attSet

    attTry = [ii for ii in attTrySet]
    errorList = []
    attTemp = []

    for iTry in attTry:
        attTemp = [] + attributeList
        attTemp.append(iTry)
        xTrainTemp = xattr_select(xListTrain, attTemp)
        xTestTemp = xattr_select(xListTest, attTemp)

        xTrain = numpy.array(xTrainTemp)
        xTest = numpy.array(xTestTemp)

        #标签是固定的  不会改变
        yTrain = numpy.array(labelsTrain)
        yTest = numpy.array(labelsTest)

        wineQModel = linear_model.LinearRegression()
        wineQModel.fit(xTrain, yTrain)

        rmsError = numpy.linalg.norm((yTest-wineQModel.predict(xTest)), 2)/sqrt(len(yTest))
        errorList.append(rmsError)
        attTemp = []

    iBest = numpy.argmin(errorList)
    attributeList.append(attTry[iBest])
    oosError.append(errorList[iBest])

namesList = [names[i] for i in attributeList]
print(namesList)

x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.figure()

print(oosError)
indexBest = oosError.index(min(oosError))
print(indexBest)

print(attributeList)
attributeBest = attributeList[1 : (indexBest+1)]

xTrainTemp = xattr_select(xListTrain, attributeBest)

xTestTemp = xattr_select(xListTest, attributeBest)
xTrain = numpy.array(xTrainTemp)
xTest = numpy.array(xTestTemp)

wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain, yTrain)
errorVector = yTest - wineQModel.predict(xTest)
print(errorVector)
plt.hist(errorVector)
plt.figure()

plt.scatter(wineQModel.predict(xTest), yTest, s=100, alpha=0.10)

plt.show()


