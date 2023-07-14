#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import pandas as pd
from sklearn import metrics
from joblib import load


st.sidebar.title('Transaction Information')

html_temp = """
<div style="background-cclor:Blue;padding:1epx">

<h2 style="color:white;text-align:center;">Fraud Detection</h2>
</div><br>"""


st.markdown(html_temp,unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Black;'> Select Your Model</h1>", unsafe_allow_html=True)


# Downloading the unscaled input CVS files
x_valid = pd.read_csv("unscaled_x_valid.csv")
x_valid = x_valid.iloc[:, 1:]  #removing the first column


selection = st.selectbox(" ", ["Logistic Regression","Random Forest", "Light GBM"] )

if selection =="Logistic Regression":
	st.write("You selected", selection, "model")
	model = pickle.load(open('logistic_regression_model.pkl','rb'))
elif selection =="Random Forest": 
	st.write("You selected", selection, "model")
	model = pickle.load(open('random_forest_model.pkl','rb'))
else:
	st.write("You selected", selection, "model")
	model = pickle.load(open('lgb_model.pkl','rb'))


v2 = st.sidebar.slider(label="TransactionAmt", min_value=0, max_value=1000, step=5)  #unscaled amount

v3 = st.sidebar.slider(label="Time of the day", min_value=0, max_value=23, step=1)  #_Hours
v4 = st.sidebar.slider(label="Day of the day (Monday=0, Sunday=6)", min_value=0, max_value=6, step=1)     #_Weekdays


# Modifying transaction Amount variable for 1 observation
x_valid = x_valid.iloc[0, :]   #selecting first row
x_valid = x_valid.drop(['TransactionAmt', '_Hours', '_Weekdays'])  #dropping transaction amount
x_valid = pd.DataFrame(x_valid).transpose()
x_valid.insert(0, 'TransactionAmt',v2)
x_valid.insert(58, '_Weekdays',v4)
x_valid.insert(59, '_Hours',v3)


# Scaling all the data
x_valid_scaled = x_valid.copy()
scaler=load('std_scaler.bin')	#importing scaler taken from feature engineering file
cols = list(x_valid.columns)
x_valid_scaled[cols] = scaler.transform(x_valid_scaled[cols])


# Fitting the model
y_pred_valid = model.predict(x_valid_scaled)

html_temp = """

<div style="background-cclor:8lack;padding:1epx">
<h2 style="color:white;text-align:center;">Fraud Detection Prediction</h2>

</div><br>"""

st.markdown("<h1 style='text-align: center; color: Black;'>Transaction Information</h1>", unsafe_allow_html=True)
st.table(x_valid)


st.subheader("Click PREDICT if configuration is OK")

prediction = y_pred_valid
if st.button("PREDICT"):
  if prediction==0:
    st.success(f'Transaction is SAFE :)')
  elif prediction==1:
    st.warning(f'ALARM! Transaction is FRAUDULENT :(')


