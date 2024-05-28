import streamlit as st
import pickle
import pandas as pd
import numpy as np
clf=pickle.load(open('bank_churn_1.pkl','rb'))
st.title("Dự đoán sự rời bỏ của khách hàng ngân hàng")
id=st.number_input("id khách",0,275056,1000)
Age = st.number_input("Tuổi", 18, 40, 20)
CreditScore = st.number_input("Điểm tín dụng", 0, 850, 600)
Balance = st.number_input("Số dư tài khoản", 0.0, 121263.62, 50000.0)
EstimatedSalary = st.number_input("Lương ước tính", 0.0,184866.69, 50000.0)
prediction = clf.predict([[id,Age,CreditScore,Balance,EstimatedSalary]])
d=pd.read_csv('train.csv')
if st.button("Dự đoán"):
    input_data = np.array([[id, Age, CreditScore, Balance, EstimatedSalary]])
    prediction = clf.predict(input_data)
    predicted_class = "Rời bỏ" if prediction[0] == 1 else "Ở lại"
    st.write(f"Dự đoán: {predicted_class}")
