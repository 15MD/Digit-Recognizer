import numpy as np
import pandas as pd
import scipy as sci
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import stats

x = pd.read_csv("//home//mohit//Music//Kaggal//digit-recognizer/train.csv")

 print(x.isnull().values.any())

# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# X = scaler.fit_transfrom(X)

array = x.values
A = array[:, 1:]
B = array[:, 0]

validation_size=0.20
seed=6
A_train,A_test,B_train,B_test=model_selection.train_test_split(A,B,test_size=validation_size,random_state=seed)


svm_model=SVC(C=0.001,kernel="linear").fit(A_train,B_train)
svm_predictions = svm_model.predict(A_test)
print("svm preditions ",svm_predictions)
accuracy = svm_model.score(A_test, B_test)
print(accuracy)
cm = confusion_matrix(B_test, svm_predictions)
print(cm)



