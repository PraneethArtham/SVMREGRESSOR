import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# PAGE CONFIG
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("Car Price Prediction using SVR")
st.write("Support Vector Regression Model")

# LOAD DATASET
data = pd.read_csv("svm_regressor/CarPrice_Assignment.csv")

# SHOW DATASET
st.subheader("Dataset")

if st.checkbox("Show Dataset"):
    st.write(data.head())

# FEATURES & TARGET
X = data[["enginesize"]].values
y = data["price"].values

# FEATURE SCALING
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)

y_scaled = scaler_y.fit_transform(
    y.reshape(-1, 1)
).flatten()  # we are reshaping into 2D arrays with one column and .flatten() is used to convert it back to 1D array

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# MODEL TRAINING
model = SVR(kernel="rbf")

model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# EVALUATION
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

st.write(f"MSE: {mse:.2f}")

st.write(f"R2 Score: {r2:.2f}")

# USER INPUT
st.subheader("Predict Car Price")

engine_size = st.slider(
    "Select Engine Size",
    int(data["enginesize"].min()),
    int(data["enginesize"].max()),
    100,
)

# PREDICTION BUTTON
if st.button("Predict Price"):
    input_data = np.array([[engine_size]])

    input_scaled = scaler_X.transform(input_data)

    prediction_scaled = model.predict(input_scaled)

    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

    st.success(f"Predicted Car Price: ${prediction[0][0]:,.2f}")

# VISUALIZATION
st.subheader("SVR Regression Graph")

fig, ax = plt.subplots()

# original data
ax.scatter(X, y, label="Actual Data")

# smooth curve
X_grid = np.arange(min(X), max(X), 1)

X_grid = X_grid.reshape((len(X_grid), 1))

X_grid_scaled = scaler_X.transform(X_grid)

y_grid_scaled = model.predict(X_grid_scaled)

y_grid = scaler_y.inverse_transform(y_grid_scaled.reshape(-1, 1))

ax.plot(X_grid, y_grid, color="red", label="SVR Curve")

ax.set_xlabel("Engine Size")
ax.set_ylabel("Car Price")
ax.set_title("Engine Size vs Car Price")

ax.legend()

st.pyplot(fig)
