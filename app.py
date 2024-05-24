import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Đọc dữ liệu và chuẩn hóa
raw_data = pd.read_csv("heart.csv")
process_data = raw_data.copy()
scaler = MinMaxScaler()
scale_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak", "slp", "thall", "cp"]
process_data[scale_cols] = scaler.fit_transform(process_data[scale_cols])

y = process_data.pop("output")
X = process_data

# Huấn luyện mô hình RandomForestClassifier
model_rf = RandomForestClassifier(max_depth=5, max_features='sqrt', criterion='gini', random_state=0, n_estimators=50)
model_rf.fit(X, y)

# Huấn luyện mô hình KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X, y)

# Huấn luyện mô hình LogisticRegression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X, y)

# Huấn luyện mô hình GaussianNB
model_nb = GaussianNB()
model_nb.fit(X, y)

# Tạo giao diện Streamlit
st.title("Heart Disease Prediction")
st.write("Nhập các thông số bệnh nhân để dự đoán nguy cơ mắc bệnh tim")

# Chọn thuật toán
algorithm = st.selectbox("Chọn thuật toán", ["Random Forest", "KNN", "Logistic Regression", "Gaussian Naïve Bayes"])

age = st.slider("Age", int(raw_data['age'].min()), int(raw_data['age'].max()), int(raw_data['age'].mean()))
sex = st.selectbox("Sex", [0, 1])
cp = st.slider("Chest Pain types", int(raw_data['cp'].min()), int(raw_data['cp'].max()), int(raw_data['cp'].mean()))
trtbps = st.slider("Resting Blood Pressure", int(raw_data['trtbps'].min()), int(raw_data['trtbps'].max()), int(raw_data['trtbps'].mean()))
chol = st.slider("Serum Cholestoral in mg/dl", int(raw_data['chol'].min()), int(raw_data['chol'].max()), int(raw_data['chol'].mean()))
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.slider("Resting Electrocardiographic results", int(raw_data['restecg'].min()), int(raw_data['restecg'].max()), int(raw_data['restecg'].mean()))
thalachh = st.slider("Maximum Heart Rate achieved", int(raw_data['thalachh'].min()), int(raw_data['thalachh'].max()), int(raw_data['thalachh'].mean()))
exng = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", float(raw_data['oldpeak'].min()), float(raw_data['oldpeak'].max()), float(raw_data['oldpeak'].mean()))
slp = st.slider("Slope of the peak exercise ST segment", int(raw_data['slp'].min()), int(raw_data['slp'].max()), int(raw_data['slp'].mean()))
caa = st.slider("Major vessels colored by flourosopy", int(raw_data['caa'].min()), int(raw_data['caa'].max()), int(raw_data['caa'].mean()))
thall = st.slider("Thal", int(raw_data['thall'].min()), int(raw_data['thall'].max()), int(raw_data['thall'].mean()))

# Chuẩn hóa đầu vào
input_data = pd.DataFrame([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]], 
                          columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])
input_data[scale_cols] = scaler.transform(input_data[scale_cols])

# Dự đoán
# Dự đoán
if st.button("Predict"):
    if algorithm == "Random Forest":
        prediction = model_rf.predict(input_data)
        prediction_proba = model_rf.predict_proba(input_data)[0]
    elif algorithm == "KNN":
        prediction = model_knn.predict(input_data)
        prediction_proba = model_knn.predict_proba(input_data)[0]
    elif algorithm == "Logistic Regression":
        prediction = model_lr.predict(input_data)
        prediction_proba = model_lr.predict_proba(input_data)[0]
    elif algorithm == "Gaussian Naïve Bayes":
        prediction = model_nb.predict(input_data)
        prediction_proba = model_nb.predict_proba(input_data)[0]

    if prediction is not None:  # Kiểm tra xem prediction đã được gán giá trị hay chưa
        st.write(f"Prediction: {'Có nguy cơ mắc bệnh tim' if prediction[0] else 'Không có nguy cơ mắc bệnh tim'}")
        st.write(f"Probability: {prediction_proba}")
    else:
        st.write("No prediction available.")
