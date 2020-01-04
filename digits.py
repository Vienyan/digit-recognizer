import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import operator

#将32*32的像素集转换为1*1024的向量
# def img2vector(filename):
#     returnVector = np.zeros((1,1024))
#     fr = open(filename)
#     for i in range(32):
#         lineStr = fr.readline()
#         #print(lineStr)
#         for j in range(32):
#             returnVector[0,32*i+j] = int(lineStr[j])
#     return returnVector

#将28*28的像素矩阵转换为1*784的向量
def img2vector(juzheng):
    returnVector = np.zeros((1,784))
    for i in range(28):
        for j in range(28):
            returnVector[0,28*i+j] = juzheng[i][j]
    return returnVector

#
# imgVector = img2vector('testDigits/7_0.txt')
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y_train = train['label']
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

#打印出图片来看看
# fig = plt.figure()
# plt.imshow(X_train[2].reshape((28,28)),cmap=plt.cm.binary)
# plt.show()

#k近邻算法
#inX为需要分类的数据，dataSet为训练数据，labels为训练数据的标签，k为k的取值
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


#拼成一个训练集合
trainingMat = np.zeros((42000,784))
for i in range(42000):
    trainingMat[i, :] = img2vector(X_train[i])

resultArray = []
for i in range(28000):
    classifierResult = classify0(img2vector(X_test[i]),  trainingMat,Y_train,3)
    resultArray.append(classifierResult)
    print("the classifier came back with:%d" % classifierResult)

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), pd.Series(resultArray,name="Label")],axis = 1)
submission.to_csv("mykNN.csv",index=False)

print(1)
#classifierResult=classify0(X_test[1],X_train,Y_train,3)

# def handwritingClassTest():
#     hwLabels = []
#     trainingFileList = os.listdir('trainingDigits')
#     m = len(trainingFileList)
#     trainingMat = np.zeros((m,1024))
#     for i in range(m):
#         fileNameStr = trainingFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         classNumStr = int(fileStr.split('_')[0])
#         hwLabels.append(classNumStr)
#         trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
#     testFileList = os.listdir('testDigits')
#     errorCount = 0.0
#     mTest = len(testFileList)
#     for i in range(mTest):
#         fileNameStr = testFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         classNumStr = int(fileStr.split('_')[0])
#         vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
#         classifierResult = classify0(vectorUnderTest,\
#                                      trainingMat,hwLabels,3)
#         print("the classifier came back with:%d, the real answer is :%d" % (classifierResult,classNumStr) )
#         if(classifierResult != classNumStr) :errorCount +=1.0
#     print("total number of error is %d" % errorCount)
#     print("total error rate is %f" % (errorCount/float(mTest)))

#handwritingClassTest()






