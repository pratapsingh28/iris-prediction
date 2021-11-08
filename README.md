# iris-prediction
#iris dataset flower prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs

from sklearn.datasets import load_iris

ld=load_iris()

dir(ld)

df=pd.DataFrame(data=ld.data,columns=ld.feature_names)

df.head()

df["output"]=pd.DataFrame(ld.target)

df.head()

df.output.unique()

df.shape

from sklearn.model_selection import train_test_split
x=df.drop(["output"],axis=1)
y=df[["output"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3)

from sklearn.svm  import SVC

sv=SVC()

sv.fit(x_train,y_train)

sv.score(x_train,y_train)

plt.scatter(df["sepal length (cm)"],df["sepal width (cm)"],color="Green")

ypred=sv.predict(x_test)

ypred[0:5]

x_test.head()

y_test.head()



