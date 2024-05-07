# EX-06 Implementation of Decision Tree Classifier Model for Predicting Employee Churn
# DATE:
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Paul Samson.S
RegisterNumber: 212222230104
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## DATASET
![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/eadd9524-ca49-4702-a038-1b17d643e32e)

## data.info()
![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/bd66b8e0-85da-4b65-8cec-ccd1ab3cfa92)


## CHECKING IF NULL VALUES ARE PRESENT
![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/7c01ab07-be7f-4aef-a166-f750c7e9bc3e)

## VALUE_COUNTS()
![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/66008e54-d055-4478-a980-c4735a723a24)


## DATASET AFTER ENCODING

![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/b2ce6ba2-529c-4b17-8960-fad60d7a2c6d)
## X-VALUES

![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/1f833ca6-e8e3-4766-ae7b-e914b5081aa9)


## ACCURACY
![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/a0dfbbcb-70f0-4043-b092-1d87747e66d8)

## dt.predict()
![image](https://github.com/haritha-venkat/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121285701/02864350-6c13-4d56-b494-53407665fd4e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
