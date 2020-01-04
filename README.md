# digit-recognizer
kaggle入门竞赛--Digit Recognizer的个人代码

比赛地址：https://www.kaggle.com/c/digit-recognizer

该竞赛的目的是根据训练数据集中的手写数字灰度图像，对测试集的手写数字图像进行识别。

每个图像的高度为28像素，宽度为28像素，总计784像素，像素值是0到255之间的整数（含）。

本人目前基础还比较差，主要是直接使用k近邻算法进行分类，最终识别准确率为0.96942，   
在其他参赛选手面前相形见绌，不过这是我第一次自己做这些题目，希望以后不断改进，    
能使得算法的效果不断提高


# 1.引入数据集，并对像素点做归一化处理

  train = pd.read_csv("train.csv")   
  test = pd.read_csv("test.csv")     
  Y_train = train['label']    
  X_train = train.drop(labels = ["label"],axis = 1)      //提取出训练集的标签    
  X_train = X_train / 255.0    
  X_test = test / 255.0    
  X_train = X_train.values.reshape(-1,28,28,1)       //转换为28×28的矩阵，方便后面输出图片查看    
  X_test = X_test.values.reshape(-1,28,28,1)    

 
# 2.打印出某个训练样本看看
 
 fig = plt.figure()   
 plt.imshow(X_train[2].reshape((28,28)),cmap=plt.cm.binary)   
 plt.show()   


# 3.将像素矩阵转为一维向量  
将28×28的像素矩阵转换为1×784的向量

def img2vector(matrix):   
    returnVector = np.zeros((1,784))   
    for i in range(28):    
        for j in range(28):    
            returnVector[0,28*i+j] = matrix[i][j]    
    return returnVector   

 
# 4.k近邻算法核心
inX为需要分类的数据，dataSet为训练数据，labels为训练数据的标签，k为k的取值

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


# 5.对测试集进行分类
#拼成一个训练集合

trainingMat = np.zeros((42000,784))   
for i in range(42000):    
    trainingMat[i, :] = img2vector(X_train[i])    

resultArray = []    
for i in range(28000):   
    classifierResult = classify0(img2vector(X_test[i]),  trainingMat,Y_train,3)    
    resultArray.append(classifierResult)   
    print("the classifier came back with:%d" % classifierResult)   


# 6.保存结果

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), pd.Series(resultArray,name="Label")],axis = 1)     
submission.to_csv("mykNN.csv",index=False)    
 ` ` `






