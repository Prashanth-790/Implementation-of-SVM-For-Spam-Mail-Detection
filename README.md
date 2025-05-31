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
Developed by: Abishek Priyan M
RegisterNumber: 212224240004
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
![image](https://github.com/user-attachments/assets/9d9ff09a-3791-4e7d-b87c-56dc3c4fa23c)


## Head():
![image](https://github.com/user-attachments/assets/ced0bd07-dc76-4345-954d-9cf9edca844c)


## Info():
![image](https://github.com/user-attachments/assets/8a9f9e63-d10e-44b9-874f-58b7297aa14d)


## isnull().sum():
![image](https://github.com/user-attachments/assets/d027c68c-f34b-4085-92f2-1559b11ba82a)


## Prediction of y:
![image](https://github.com/user-attachments/assets/f5fc34ee-e52f-4caa-9878-3f31fa35ec32)


## Accuracy:
![image](https://github.com/user-attachments/assets/7e0f3d74-4c81-46a3-b694-89a7cf43bd2d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
