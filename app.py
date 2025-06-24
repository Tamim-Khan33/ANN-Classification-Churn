import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import numpy as np
import pickle
import streamlit as st

"""Load the model,one hot encoding,scaling"""
model=tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)


with open('one_hot_geo.pkl','rb') as file:
    one_hot_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)



st.title("Bank Customer Churn Prediction")

# Selectbox for categorical inputs
geography = st.selectbox("Geography",one_hot_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)

# Number inputs for numerical features
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.slider("Age", min_value=18, max_value=100, value=40)
tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Account Balance", min_value=0.0, value=60000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])  # 1 = Yes, 0 = No
is_active_member = st.selectbox("Is Active Member?", [0, 1])  # 1 = Yes, 0 = No
estimated_salary = st.number_input("Estimated Salary")

input_data =pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': geography,
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# """one hot encode geo"""
geo_encoded=one_hot_geo.transform([[geography]])
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_geo.get_feature_names_out(['Geography']))


##combine new data
input_data=input_data.drop('Geography',axis=1)
input_data=pd.concat([input_data,geo_encoded_df],axis=1)

# """scaling the data"""
input_scaled=scaler.transform(input_data)

# """prediction"""
prediction=model.predict(input_scaled)
prediction_probability=prediction[0][0]
print(prediction_probability)
st.write(f"Churn Probability:{prediction_probability:.2f}")
if prediction_probability>0.5:
    st.write("the customer will churn")
else:
    st.write("the customer won't churn ")