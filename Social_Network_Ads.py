# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,1:4].values
y = dataset. iloc[:,4:5].values

#Encoding the independent variables(X)
#importing preprocessing from sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#importing the compose sub_library from sklearn
from sklearn.compose import make_column_transformer
#defining the function of the make_column_tranformer
column_changer=make_column_transformer((OneHotEncoder(),[0]),remainder='passthrough')

#assigning the variable X to the column_changer
X = column_changer.fit_transform(X)


#splitting the dataset into train data and text data
from sklearn.model_selection import train_test_split

#creating variables to store X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler

#assigning the StandardScaler to a variable sc_X
sc_X=StandardScaler()

#fitting and transforming the X_train and the X test
X_train=sc_X.fit_transform(X_train)

#fitting X_test
X_test=sc_X.fit_transform(X_test)

#training the decision tree module
from sklearn.tree import DecisionTreeRegressor
#creating a variable and assigning the Decision Tree Regession algorithm
xty_machine_module=DecisionTreeRegressor(random_state=1)

#training the module xty_machine with X_train and y_train
xty_machine_module.fit(X_train,y_train)

#making a prediction
prediction_result=xty_machine_module.predict(X_test)

prediction_result

#TODO:add comment
from sklearn.metrics import accuracy_score

#TODO: add comment
score=accuracy_score(y_test,prediction_result)
print(score*100,'%')
