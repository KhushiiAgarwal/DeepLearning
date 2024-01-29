import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
import tensorflow as tf

df=pd.read_csv("wbc_csv.csv")
df.drop(['id'],axis=1,inplace=True)
df=df.iloc[:,0:31]

print(df.shape)
df.info()
df.describe()

df.isnull().sum()

"""# One Hot Encoding"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
df['diagnosis']=df['diagnosis'].astype('category')
df['diagnosistics']=df['diagnosis'].cat.codes
enc=OneHotEncoder()
enc_df=pd.DataFrame(enc.fit_transform(df[['diagnosistics']]).toarray())

y=df['diagnosis']
x=df.drop(labels=['diagnosis','diagnosistics'], axis=1)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(x)
scaled_df = pd.DataFrame(scaled_df)
scaled_df

"""# Variance Threshold & CHI sq"""

from sklearn.feature_selection import VarianceThreshold
var_thres=VarianceThreshold(threshold=0)
var_thres.fit(scaled_df)
x.columns[var_thres.get_support()]

#Variance Threshold
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_df,y,test_size=0.3,stratify=y,random_state=50)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train.shape, X_test.shape

var_thres=VarianceThreshold(threshold=0)
var_thres.fit(X_train)
sum(var_thres.get_support())

"""#ANN: Artificial Neural Networks"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(scaled_df, y, test_size=0.15, stratify=y, random_state=26)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialising ANN
ann = tf.keras.models.Sequential()

#Adding  Hidden Layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Adding Output Layer
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

#Fitting ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

ann.evaluate(X_test,y_test,batch_size=32)

test_loss, test_accuracy = ann.evaluate(X_test, y_test, batch_size=32)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

y_pred_probs = ann.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (using sklearn): {test_accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred )
print("Confusion Matrix:\n", conf_matrix)


plt.figure(figsize=(12,7))
sn.heatmap(cm, annot= True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
