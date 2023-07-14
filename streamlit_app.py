#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
from sklearn import metrics



st.sidebar.title('Transaction Information')

html_temp = """
<div style="background-cclor:Blue;padding:1epx">

<h2 style="color:white;text-align:center;">Fraud Detection</h2>
</div><br>"""


st.markdown(html_temp,unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Black;'> Select Your Model</h1>", unsafe_allow_html=True)

## upload CVS files
x_valid = pd.read_csv("X_valid.csv")
x_valid = x_valid.iloc[:, 1:]
y_valid = pd.read_csv("y_valid.csv", usecols=range(1,2))

selection = st.selectbox("",["Logistic Regression","Random Forest", "Light GBM"])

if selection =="Logistic Regression":
	st.write("You selected", selection, "model")
	model = pickle.load(open('logistic_regression_model.pkl','rb'))
elif selection =="Random Forest": 
	st.write("you selected”, selection, “model")
	model = pickle.load(open('random_forest_model.pkl','rb'))
else:
	st.write("you selected”, selection, “model")
	model = pickle.load(open('light_gbm_model.pkl','rb'))
	


v2 = st.sidebar.slider(label="TransactionAmt", min_value=-10.0, max_value=15.80, step=0.1)

x_default = x_valid.iloc[0, :]  #dropping transaction amount
x_default = x_default.drop('TransactionAmt')
x_default = pd.DataFrame(x_default).transpose()
x_default.insert(0, 'TransactionAmt',v2)

y_pred_valid_prob = model.predict_proba(x_default)[:, 1]
y_pred_valid = model.predict(x_default)

html_temp = """

<div style="background-cclor:8lack;padding:1epx">
<h2 style="color:white;text-align:center;">Fraud Detection Prediction</h2>

</div><br>"""

st.markdown("<h1 style='text-align: center; color: Black;'>Transaction Information</h1>", unsafe_allow_html=True)
st.table(x_default)


st.subheader("Click PREDICT if configuration is OK")

prediction = y_pred_valid
if st.button("PREDICT"):
  if prediction==0:
    st.success(f'Transaction is SAFE :)')
  elif prediction==1:
    st.warning(f'ALARM! Transaction is FRAUDULENT :(')


