import streamlit as st
import pandas as pd
import joblib

# Load mô hình
model = joblib.load("tree_model.pkl")

# Tiêu đề
st.title("Dự đoán chi phí bảo hiểm y tế")

# Giao diện người dùng
age = st.slider("Tuổi", 18, 65, 30)
sex = st.selectbox("Giới tính", ["male", "female"])
bmi = st.number_input("Chỉ số BMI", 15.0, 45.0, 25.0)
children = st.slider("Số con", 0, 5, 0)
smoker = st.radio("Hút thuốc?", ["yes", "no"])
region = st.selectbox("Khu vực", ["northeast", "northwest", "southeast", "southwest"])

# Tiền xử lý dữ liệu đầu vào
def preprocess():
    return pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0],
    })

# Dự đoán khi người dùng nhấn nút
if st.button("Dự đoán chi phí"):
    X_input = preprocess()
    prediction = model.predict(X_input)[0]
    st.success(f"Chi phí bảo hiểm ước tính: {prediction:,.2f} USD")

