import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM, DDMConfig
from river.datasets import synth
import altair as alt

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
detection_delay = 0.0

# Show default parameter values on the left side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameter Values")
    st.write(f"Drift Detector: DDM")
    # Dropdown for selecting the regression model
    model_choice = st.selectbox(
        "Select the regression model:",
        ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor")
    )

    # Sidebar for ADWIN configuration
    warning_level = st.slider("Warning Level", 0.0, 0.1, 0.01)
    drift_level = st.slider("Drift Level", 1.0, 10.0, 3.0)
    min_num_instances = st.slider("Min Number of Instances", 1, 5000, 1000)
    
    # Input fields for drift points, step set to 1000
    first_point = st.number_input("Enter the first drift point:", min_value=1000, value=7000, step=1000)
    second_point = st.number_input("Enter the second drift point:", min_value=first_point+1000, value=11000, step=1000)

# Button to trigger computation
if st.button("Check for Drift"):
    # Dataset generation
    dataset = synth.FriedmanDrift(
        drift_type='gra',
        position=(first_point, second_point),
        seed=42
    )

    # Initialize the data containers
    data = []
    for i, (x, y) in enumerate(dataset):
        x_values = list(x.values())
        data.append(x_values + [y])
        if i >= second_point:  # Limiting to second_point samples for simplicity
            break

    # Define the column names and create the DataFrame
    column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
    df = pd.DataFrame(data, columns=column_names)

    train = df.iloc[:(first_point//2)]
    stream = df.iloc[(first_point//2):]

    X_train = train.drop(columns='y').values
    y_train = train['y'].values

    # Select the regression model based on the dropdown choice
    if model_choice == "Linear Regressor":
        model = LinearRegression()
    elif model_choice == "SVM Regressor":
        model = SVR()
    elif model_choice == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
    else:
        model = RandomForestRegressor()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipeline.fit(X=X_train, y=y_train)

    odp = [first_point]  # Track the actual point where drift occurs

    X_stream = stream.drop(columns='y').values
    y_stream = stream['y'].values

    # Detector configuration and instantiation
    config = DDMConfig(
        warning_level=warning_level,  # Define the warning and drift levels
        drift_level=drift_level,
        min_num_instances=min_num_instances,
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
    detection_delay =(detection_delays[0]) if detection_delays else None

    # Update parameter values with actual results
    with col1:
        st.write(f"False Alarms: {false_alarms}")
        st.write(f"False Alarm Rate: {false_alarm_rate}")
        st.write(f"Average Detection Delay: {detection_delay}")

    # Prepare the data for Altair visualizations
    error_data = pd.DataFrame({
        'Index': np.arange(len(errors))+(first_point//2),
        'Mean Squared Error': errors
    })

    drift_data = pd.DataFrame({
        'Index': np.arange(len(data)),
        'Detected Drift Indicator': [1 if i in detected_drifts else 0 for i in range(len(data))],
        'Actual Drift Indicator': [1 if i >= odp[0] else 0 for i in range(len(data))]
    })

    # Right side (graphs)
    with col2:
        st.subheader('Mean Squared Error Over Data Points')
        mse_chart = alt.Chart(error_data).mark_line(color='purple').encode(
            x='Index',
            y='Mean Squared Error'
        ).properties(
            width=600,
            height=300,
            title="Behavior of Mean Squared Error Over Data Points"
        )
        st.altair_chart(mse_chart, use_container_width=True)

        st.subheader('Drift Detection Indicator')
        drift_chart = alt.Chart(drift_data).mark_line().transform_fold(
            ['Detected Drift Indicator', 'Actual Drift Indicator'],
            as_=['Indicator Type', 'Value']
        ).encode(
            x='Index',
            y=alt.Y('Value:Q', axis=alt.Axis(title='Drift Detection (0 or 1)')),
            color='Indicator Type:N'
        ).properties(
            width=600,
            height=300,
            title="Drift Detection Indicator"
        )
        st.altair_chart(drift_chart, use_container_width=True)
