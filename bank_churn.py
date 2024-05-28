import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle as pickle
import streamlit as st
import numpy as np

d = pd.read_csv('train.csv')

feature = ['id', 'Age', 'CreditScore', 'Balance', 'EstimatedSalary']
target = ['Exited']
X = d[feature]
y = d[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train.values.ravel())

print(clf.score(X_test, y_test))

print("Saving model to pickle file.")
pickle.dump(clf, open("bank_churn_1.pkl", 'wb'))
