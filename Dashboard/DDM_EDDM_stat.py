import streamlit as st
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from river.datasets import synth
from frouros.detectors.concept_drift.streaming.window_based.adwin import ADWIN
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM, DDMConfig
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM, EDDMConfig

# Initialize Streamlit app
st.title("Concept Drift Detection with Streamlit")

# Sidebar for user inputs
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Choose Regression Model", ("Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"))

st.sidebar.header("Drift Detection Configuration")
drift_method = st.sidebar.selectbox("Drift Detection Method", ("DDM", "EDDM"))
warning_level = st.sidebar.slider("Warning Level", 0.5, 5.0, 2.0)
drift_level = st.sidebar.slider("Drift Level", 0.5, 5.0, 3.0)
min_num_instances = st.sidebar.slider("Min Number of Instances", 100, 1000, 30)
quantile_threshold = st.sidebar.slider("Quantile Threshold", 0.9, 1.0, 0.95, step=0.01)

# Sidebar for dataset parameters
st.sidebar.header("Dataset Parameters")
dp1 = st.sidebar.number_input("Drift Start Position (dp1)", min_value=100, max_value=10000, value=2000)
dp2 = st.sidebar.number_input("Drift End Position (dp2)", min_value=100, max_value=10000, value=3000)
split_index = st.sidebar.number_input("Train/Stream Split Index", min_value=100, max_value=10000, value=1000)

# Main page to display results
st.header("Drift Detection Results")

@st.cache
def initialize_data(dp1, dp2, seed=42):
    dataset = synth.FriedmanDrift(drift_type='gra', position=(dp1, dp2), seed=seed)
    data = []
    for i, (x, y) in enumerate(dataset):
        x_values = list(x.values())
        data.append(x_values + [y])
        if i >= dp2:
            break
    column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
    df = pd.DataFrame(data, columns=column_names)
    return df

@st.cache
def split_data(df, split):
    train = df.iloc[:split]
    stream = df.iloc[split:]
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    return train, val, stream

def train_model(train, model):
    X_train = train.drop(columns='y').values
    y_train = train['y'].values
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline

def calculate_threshold(pipeline, val, quantile_threshold):
    X_val = val.drop(columns='y').values
    y_val = val['y'].values
    y_pred = pipeline.predict(X_val)
    errors_val = (y_val - y_pred) ** 2
    threshold = np.quantile(errors_val, quantile_threshold)
    return threshold

def setup_ddm(warning_level, drift_level, min_num_instances):
    config = DDMConfig(warning_level=warning_level, drift_level=drift_level, min_num_instances=min_num_instances)
    return DDM(config=config)

def setup_eddm(alpha, beta, level, min_num_misclassified_instances):
    config = EDDMConfig(alpha=alpha, beta=beta, level=level, min_num_misclassified_instances=min_num_misclassified_instances)
    return EDDM(config=config)

def process_stream(X, y, pipeline, detector, threshold, odp):
    detected_drifts = []
    false_alarms = 0
    detection_delays = []
    y_preds = []
    errors = []

    for i in range(len(X)):
        X_i = X[i].reshape(1, -1)
        y_i = y[i].reshape(1, -1)
        y_pred = pipeline.predict(X_i)
        y_preds.append(y_pred[0])
        error = mean_squared_error(y_i, y_pred)
        errors.append(error)
        binary_error = 1 if error > threshold else 0
        detector.update(value=binary_error)

        if detector.drift:
            detected_drifts.append(i + len(train))
            if i + len(train) < odp:
                false_alarms += 1
            else:
                detection_delays.append((i + len(train)) - odp)

    false_alarm_rate = (false_alarms / (len(X) - odp)) * 100 if detected_drifts else 0
    detection_delay = detection_delays[0] if detection_delays else None
    return false_alarm_rate, detection_delay, errors, detected_drifts

def plot_errors(errors: List[float]) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(errors)), errors, label="MSE Error", color='purple')
    plt.xlabel("Instance Index")
    plt.ylabel("Mean Squared Error")
    plt.title("Behavior of Mean Squared Error Over Data Points")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_drift_indicators(data_length: int, detected_drifts: List[int], odp: int):
    drift_indicator = np.zeros(data_length)
    for drift_point in detected_drifts:
        if drift_point < data_length:
            drift_indicator[drift_point] = 1

    odp_indicator = np.zeros(data_length)
    for i in range(odp, data_length):
        odp_indicator[i] = 1

    plt.figure(figsize=(12, 4))
    plt.plot(drift_indicator, label='Detected Drift Indicator', color='red', drawstyle='steps-post')
    plt.plot(odp_indicator, label='Actual Drift Indicator (ODP)', color='green', linestyle='-', drawstyle='steps-post')
    plt.xlabel('Index')
    plt.ylabel('Drift Detection (0 or 1)')
    plt.title('Drift Detection Indicator')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Initialize dataset
df = initialize_data(dp1, dp2, seed=42)

# Split dataset into train, validation, and stream
train, val, stream = split_data(df, split_index)

# Select model based on user input
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Decision Tree Regressor":
    model = DecisionTreeRegressor()
elif model_choice == "Random Forest Regressor":
    model = RandomForestRegressor()
else:
    model = SVR()

# Train the model
pipeline = train_model(train, model)

# Calculate the quantile-based threshold
threshold = calculate_threshold(pipeline, val, quantile_threshold)

# Set up the chosen drift detector
if drift_method == "DDM":
    detector = setup_ddm(warning_level, drift_level, min_num_instances)
else:
    detector = setup_eddm(alpha=0.95, beta=0.9, level=2.0, min_num_misclassified_instances=min_num_instances)

# Process stream data and detect drifts
X = stream.drop(columns='y').values
y = stream['y'].values
false_alarm_rate, detection_delay, errors, detected_drifts = process_stream(X, y, pipeline, detector, threshold, odp=dp1)

# Display results
st.subheader("Detection Results")
st.write(f"False Alarm Rate: {false_alarm_rate:.2f}%")
st.write(f"Detection Delay: {detection_delay}")
st.write(f"Detected Drifts: {detected_drifts}")

# Plot error and drift indicators
plot_errors(errors)
plot_drift_indicators(dp2, detected_drifts, odp=dp1)
