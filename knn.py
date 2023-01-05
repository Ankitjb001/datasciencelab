from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
# d2=pd.read_csv("san.csv")
dataset=load_iris()
# # a=d2.gpa
# # b=d2.year
# # print(dataset)

a=dataset.data
# print(aa)
b=dataset.target
# print(ba)
a_train,a_test,b_train,b_test=train_test_split(a,b,train_size=0.3,random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(a_train,b_train)
pred=knn.predict(a_test)
acc=accuracy_score(b_test,pred)
print(pred)
print(acc)