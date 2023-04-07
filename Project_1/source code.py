import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense,Dropout
from tensorflow.keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split

train=pd.read_csv('../input/spaceship-titanic/train.csv')
test=pd.read_csv('../input/spaceship-titanic/test.csv')
train = train.drop("Name",axis = 1) #去掉影響較小的欄位
test = test.drop("Name",axis = 1)
train["Transported"] = train["Transported"].astype(int)

for col in train.columns:
    if(train[col].isna().sum() !=0 and train[col].dtype != "object"):
        train.loc[train[col].isnull() == True,col] = train[col].median()
    else:
        train.loc[train[col].isnull() == True,col] = "miss"
for col in test.columns:
    if(test[col].isna().sum() !=0 and test[col].dtype != "object"):
        test.loc[test[col].isnull() == True,col] = test[col].median()
    else:
        test.loc[test[col].isnull() == True,col] = "miss"
    
train["Transported"] = train["Transported"].astype(int)
train["VIP"] = train["VIP"].astype(str)
train["CryoSleep"] = train["CryoSleep"].astype(str)
test["VIP"] = test["VIP"].astype(str)
test["CryoSleep"] = test["CryoSleep"].astype(str)#改資料型別
f = []
scaler = StandardScaler()
for col in train.columns:
    if(train[col].dtypes == "float64"):
        f.append(col)
train[f] = scaler.fit_transform(train[f])
test[f] = scaler.transform(test[f])#用平均值和標準差縮小特徵值的範圍

labelencoder = LabelEncoder()
for col in train.columns:
    if(train[col].dtypes == "object" and col != "PassengerId"):
        train[col] = labelencoder.fit_transform(train[col])
        test[col] = labelencoder.fit_transform(test[col])
t = []
t.append("Cabin")
train[t] = scaler.fit_transform(train[t])
test[t] = scaler.fit_transform(test[t])#透過labelencoder把物件改成數字讓model能讀進去

x = train.drop(["PassengerId","Transported"] , axis = 1)
y = train["Transported"]

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2,random_state=54088,stratify=y)

x = train.drop(["PassengerId","Transported"] , axis = 1)
y = train["Transported"]

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2,random_state=54088,stratify=y)
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu", name="L1",input_shape = (len(x.columns),) ),
        layers.Dense(32,activation="relu", name="L2"),
        layers.Dropout(rate = 0.2),
        layers.Dense(64,activation="relu", name="L3"),
        layers.Dropout(rate = 0.2),
        layers.Dense(64,activation="relu", name="L4"),
        layers.Dropout(rate = 0.1),
        layers.Dense(1,activation="sigmoid", name="output"),
        
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
my_callback = [tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10)]
history = model.fit(
    x_train, 
    y_train,
    callbacks = [my_callback],
    batch_size=36, 
    epochs=100, 
    validation_data=(x_test, y_test)
)

x_test = test.drop(["PassengerId"],axis=1)
sub = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")
sub["Transported"] = (model.predict(x_test) > 0.5).astype(bool)
sub.to_csv("submission.csv",index=False)
