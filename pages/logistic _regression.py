import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import streamlit as st 
import plotly.express as px
from PIL import Image
import numpy as np
from sklearn import svm #contain all the clasifications that we will used, Only accept numbers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statistics import mean, median, variance, stdev
from openpyxl import load_workbook # to modify the data in excel
from app import loan_data_conv, X, Y
from sklearn.naive_bayes import CategoricalNB, GaussianNB  # to deal with the catigory values
from imblearn.over_sampling import SMOTE   #مكتبة موازنة الصفوف
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore') 

df = loan_data_conv

st.subheader("Train to test Split")  # Done
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)  
st.write("X_shape: ",X.shape)
st.write("\n  X_train.shape: ", X_train.shape)
st.write("\n  X_test.shape: ", X_test.shape)

X = loan_data_conv.drop(columns=['Loan_Status'],axis=1)
Y = loan_data_conv['Loan_Status']


st.subheader("Gender & Loan Status")
fig = plt.figure(figsize=(4, 2))
sns.countplot(data=loan_data_conv, x='Gender',hue='Loan_Status',palette=["#fc9272","#fee0ff"])
st.pyplot(fig)

st.subheader("Married & Loan Status")
fig = plt.figure(figsize=(4, 2))
sns.countplot(data=loan_data_conv, x='Married',hue='Loan_Status',palette=["#fc9272","#fee0ff"])
st.pyplot(fig)


st.subheader("education & Loan Status")
fig = plt.figure(figsize=(4, 2))
sns.countplot(data=loan_data_conv, x='Education',hue='Loan_Status',palette=["#fc9272","#fee0ff"])
st.pyplot(fig)

st.subheader("Self_Employed & Loan Status")
fig = plt.figure(figsize=(4, 2))
sns.countplot(data=loan_data_conv, x='Self_Employed',hue='Loan_Status',palette=["#fc9272","#fee0ff"])
st.pyplot(fig)


st.subheader("education & Loan Status")
fig = plt.figure(figsize=(4, 2))
sns.countplot(data=loan_data_conv, x='Education',hue='Loan_Status',palette=["#fc9272","#fee0ff"])
st.pyplot(fig)

model = LogisticRegression()
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

st.write('confusion_matrix')
#رسم مصفوفة الارتباك
fig = plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True)
st.pyplot(fig)


st.write('Logistic Regression accuracy = ', metrics.accuracy_score(Y_pred,Y_test))

st.write("Y_predicted",Y_pred)
st.write("Y_test",Y_test)


lr_prediction = {'" lr_prediction result "': [100 * accuracy_score(Y_test,Y_pred),
                    100 * precision_score(Y_test,Y_pred,average='macro'),
                    100 * recall_score(Y_test,Y_pred,average='macro'),
                    100*f1_score(Y_test,Y_pred,average='macro')]}

lr_prediction1 = {'': ['accuracy_score',
            'precision_score',
            'Recall_score',
            'F1_score'],
     '" lr_prediction result "': [100 * accuracy_score(Y_test,Y_pred),
                    100 * precision_score(Y_test,Y_pred,average='macro'),
                    100 * recall_score(Y_test,Y_pred,average='macro'),
                    100*f1_score(Y_test,Y_pred,average='macro')]}

st.write(pd.DataFrame(data=lr_prediction1))
