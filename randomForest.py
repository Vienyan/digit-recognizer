from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#trainData = pd.read_csv('train.csv',nrows=100)
trainData = pd.read_csv('train.csv')

y = trainData['label']     #标签
x = trainData.drop(['label'], axis = 1)  #训练样本
#x_test = pd.read_csv('test.csv',nrows=20)
x_test = pd.read_csv('test.csv',skiprows=24000,nrows=4000) 

model= RandomForestClassifier(n_estimators=1000)
model.fit(x, y)


predicted= model.predict(x_test)
print(predicted)

name = ['ImageId','Label']
ImageId=[i+1 for i in range(len(x_test))]

data = {'ImageId':ImageId,'Label':predicted}
conserve = pd.DataFrame(data)
conserve.to_csv('result.csv',index=None, mode='a', header=False)
#conserve.to_csv('result.csv',index=None)