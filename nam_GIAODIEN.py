import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC

#doc du lieu file csv
data = pd.read_csv('data/mushrooms.csv')
#data.head()
#data.describe()

y = data["class"].to_frame()
y.head()

#xoa cot class
X = data.drop("class", axis=1)
X.head()

#su dung thu vien defaultdict dat gia tri mat dinh trong [tu dien]
tudien = defaultdict(LabelEncoder)
X = X.apply(lambda x: tudien[x.name].fit_transform(x))
X.head()

#chuyen doi sang nhan so
nhanso_y = LabelEncoder()
y = y.apply(lambda x: nhanso_y.fit_transform(x))
#gep cac nhan lai voi nhau 
le_name_mapping = dict(zip(nhanso_y.classes_, nhanso_y.transform(nhanso_y.classes_)))

print(le_name_mapping)

#open file mushrooms attributes
with open('data/attributes_enc.sav', 'wb') as f:
    pickle.dump(tudien, f)
#open file mushrooms labels
with open('data/labels_enc.sav', 'wb') as f:
    pickle.dump(nhanso_y, f)
    
#xay dung model KNN , DecisionTreeClassifier, GradientBoostingClassifier
models = []
models.append(('KNN', KNeighborsClassifier(3)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GBC', GradientBoostingClassifier(n_estimators=500, learning_rate=0.5)))


results = []
names = []

#xuat model
def exportModel(model, name="model"):
    filename = f'{name}.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(f"Model exported as: {filename}")
    

for name, model in models:
    kfold = model_selection.StratifiedKFold(n_splits=10)
    cv_results = []
    maxx = 0
    for train_index, test_index in kfold.split(X, y):
        # print(train_index, test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_results.append(recall)
        print(f"{name} - Recall: {str(recall)}, Acc: {str(accuracy)}, F1: {str(f1)}")
        if maxx < recall:
            maxx = recall
            exportModel(model, name)
            print(confusion_matrix(y_test, y_pred, [0,1]))
    cv_results = np.array(cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Mean: %f (Std: %f) Max: %f" % (name, cv_results.mean(), cv_results.std(), cv_results.max())
    print(msg)
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)

plt.show()