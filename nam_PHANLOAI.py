import numpy as np
import pandas as pd
#ve cac bieu do thong ke
import seaborn as sns
import matplotlib.pyplot as plt

#doc du lieu
data = pd.read_csv("mushrooms.csv")
data.head()
data.info()

#xem gia tri trong cot class
data["class"].unique()

#xac dinh cac gia tri 1 =p  va e = 0
data["class"] = [1 if i == "p" else 0 for i in data["class"]]

#xoa du lieu thieu stalk-root, va cot veil-type vi du lieu chi co thuoc tinh p
data.drop("veil-type",axis=1,inplace=True)
data.drop('stalk-root',axis=1,inplace=True)

for column in data.drop(["class"], axis=1).columns:
    value = 0
    step = 1/(len(data[column].unique())-1)
    for i in data[column].unique():
        data[column] = [value if letter == i else letter for letter in data[column]]
        value += step
        
        
#data_check = data.head()
#kiem tra them cac dong cuoi
data_check = data_check.append(data.tail())
data_check

from sklearn.model_selection import train_test_split
y = data["class"].values   
x = data.drop(["class"], axis=1).values    
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.3)

#Accuracy LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train,y_train)
print("Test Accuracy LogisticRegression: {}%".format(round(lr.score(x_test,y_test)*100,2)))

#Accuracy DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Test Accuracy DecisionTreeClassifier: {}%".format(round(dt.score(x_test,y_test)*100,2)))

#Accuracy RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)
print("Test Accuracy RandomForestClassifier: {}%".format(round(rf.score(x_test,y_test)*100,2)))

# LogisticRegression
from sklearn.metrics import confusion_matrix
y_pred_lr = lr.predict(x_test)
y_true_lr = y_test
cm = confusion_matrix(y_true_lr, y_pred_lr)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="white",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_lr")
plt.ylabel("y_true_lr")
plt.show()

#RandomForestClassifier
y_pred_rf = rf.predict(x_test)
y_true_rf = y_test
cm = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="white",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_rf")
plt.ylabel("y_true_rf")
plt.show()

