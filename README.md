# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: LAKSHMEN PRASHANTH R
RegisterNumber: 212224230137
```
```py
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Encoding:
![image](https://github.com/user-attachments/assets/fe2c688c-9581-4f08-a99d-e2937431f17b)


## Head():
![image](https://github.com/user-attachments/assets/3ade3e6d-52ec-45cf-a483-aa412b1847c0)


## Info():
![image](https://github.com/user-attachments/assets/a7373424-cd84-498d-9934-effda265f065)


## isnull().sum():
![image](https://github.com/user-attachments/assets/8a4d297d-37b9-4622-bfdf-72b7628e0def)


## Prediction of y:
![image](https://github.com/user-attachments/assets/f397990d-6a8d-4f59-a820-7f62072df18c)


## Accuracy:
![image](https://github.com/user-attachments/assets/b366d118-f3d7-4e6b-90aa-e50029ef3869)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
