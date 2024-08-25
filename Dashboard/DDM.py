import streamlit as st
import altair as alt
from river import datasets, linear_model, tree, drift, metrics
from river.datasets import synth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM, DDMConfig

# Title and description
st.title('Concept Drift Detection Using DDM in a Synthetic Dataset')
st.write("""
This app simulates concept drift detection using the DDM method on a synthetic dataset.
It also displays the performance metrics and drift indicators.
""")

# Dataset generation
dataset = synth.FriedmanDrift(
    drift_type='gra',
    position=(6000, 10000),
    seed=42
)

# Initialize the data containers
data = []
for i, (x, y) in enumerate(dataset):
    x_values = list(x.values())
    data.append(x_values + [y])
    if i >= 10000:  # Limiting to 10000 samples for simplicity
        break

# Define the column names and create the DataFrame
column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
df = pd.DataFrame(data, columns=column_names)

train = df.iloc[:3000]
stream = df.iloc[3000:]

X_train = train.drop(columns='y').values
y_train = train['y'].values

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]
)
pipeline.fit(X=X_train, y=y_train)

odp = [5000]  # Track the actual point where drift occurs

X_stream = stream.drop(columns='y').values
y_stream = stream['y'].values

# Detector configuration and instantiation
config = DDMConfig(
    warning_level=0.01,  # Define the warning and drift levels
    drift_level=3.0,
    min_num_instances=1000,
)
detector = DDM(config=config)

detected_drifts = []
false_alarms = 0
detection_delays = []
y_preds = []
errors = []

warning_flag = False
for i in range(len(X_stream)):
    X_i = X_stream[i].reshape(1, -1)
    y_i = y_stream[i].reshape(1, -1)
    
    # Predict and calculate the error
    y_pred = pipeline.predict(X_i)
    y_preds.append(y_pred[0])
    error = mean_squared_error(y_i, y_pred)
    errors.append(error)
    
    binary_error = 1 if error > 50.0 else 0
    
    # Update DDM with the error 
    detector.update(value=binary_error)
    
    # Check for detected drift
    if detector.drift:
        detected_drifts.append(i + len(train))  # Adjust index to match the full dataset
        
        # Determine if it's a false alarm or not
        if i + len(train) < odp[0]:
            false_alarms += 1
        else:
            detection_delays.append((i + len(train)) - odp[0])
    
    # Check for detected warning
    if not warning_flag and detector.warning:
        st.write(f"Warning detected at step {i}")
        warning_flag = True
        
false_alarm_rate = false_alarms / len(detected_drifts) if detected_drifts else 0
average_detection_delay = np.mean(detection_delays) if detection_delays else None

# Store the parameters in session state
st.session_state.table_data = {
    "Drift Detector": ["DDM"],
    "False Alarms": [false_alarms],
    "False Alarms Rate": [false_alarm_rate],
    "Drift Detection Delay": [average_detection_delay]
}

# Display results
st.write(f"False alarms: {false_alarms}")
st.write(f"False alarm rate: {false_alarm_rate}")
st.write(f"Average detection delay: {average_detection_delay}")

# Plotting the error values
st.subheader('Mean Squared Error Over Data Points')
plt.figure(figsize=(14, 7))
plt.plot(range(len(errors)), errors, label="MSE Error", color='purple')
plt.xlabel("Instance Index")
plt.ylabel("Mean Squared Error")
plt.title("Behavior of Mean Squared Error Over Data Points")
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Create binary arrays for drift detection and odp
drift_indicator = np.zeros(len(data))
for drift_point in detected_drifts:
    drift_indicator[drift_point] = 1

odp_indicator = np.zeros(len(data))
for i in range(odp[0], len(data)):
    odp_indicator[i] = 1

# Plotting the drift indicator and odp indicator
st.subheader('Drift Detection Indicator')
plt.figure(figsize=(12, 4))
plt.plot(drift_indicator, label='Detected Drift Indicator', color='red', drawstyle='steps-post')
plt.plot(odp_indicator, label='Actual Drift Indicator (ODP)', color='green', linestyle='-', drawstyle='steps-post')
plt.xlabel('Index')
plt.ylabel('Drift Detection (0 or 1)')
plt.title('Drift Detection Indicator')
plt.legend()
plt.grid(True)
st.pyplot(plt)
