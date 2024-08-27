import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM, DDMConfig
from river.datasets import synth

# Set the layout to wide mode
st.set_page_config(layout="wide")

# Title and description
st.title('Concept Drift Detection Using DDM in a Synthetic Dataset')
st.write("""
This app simulates concept drift detection using the DDM method on a synthetic dataset.
It also displays the performance metrics and drift indicators.
""")

# Default values for metrics
false_alarms = 0
false_alarm_rate = 0.0
average_detection_delay = 0.0

# Show default parameter values on the left side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameter Values")
    st.write(f"Drift Detector: DDM")
    # Input fields for drift points, step set to 1000
    first_point = st.number_input("Enter the first drift point:", min_value=1000, value=7000, step=1000)
    second_point = st.number_input("Enter the second drift point:", min_value=first_point+1000, value=11000, step=1000)
    transition_window = st.number_input("Enter the transition window size:", min_value=1000, value=2000, step=1000)
    st.write(f"False Alarms: {false_alarms}")
    st.write(f"False Alarm Rate: {false_alarm_rate}")
    st.write(f"Average Detection Delay: {average_detection_delay}")

# Button to trigger computation
if st.button("Check for Drift"):
    # Dataset generation
    dataset = synth.FriedmanDrift(
        drift_type='gsg',
        position=(first_point, second_point),
        transition_window=transition_window,
        seed=42
    )

    # Initialize the data containers
    data = []
    for i, (x, y) in enumerate(dataset):
        x_values = list(x.values())
        data.append(x_values + [y])
        if i >= second_point + transition_window:  # Limiting to end of transition window
            break

    # Define the column names and create the DataFrame
    column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
    df = pd.DataFrame(data, columns=column_names)

    train = df.iloc[:(first_point//2)]
    stream = df.iloc[(first_point//2):]

    X_train = train.drop(columns='y').values
    y_train = train['y'].values

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X=X_train, y=y_train)

    odp = [first_point]  # Track the actual point where drift occurs

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

    false_alarm_rate = false_alarms / len(detected_drifts) if detected_drifts else 0
    average_detection_delay = np.mean(detection_delays) if detection_delays else None

    # Update parameter values with actual results
    with col1:
        st.write(f"False Alarms: {false_alarms}")
        st.write(f"False Alarm Rate: {false_alarm_rate}")
        st.write(f"Average Detection Delay: {average_detection_delay}")

    # Right side (graphs)
    with col2:
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
