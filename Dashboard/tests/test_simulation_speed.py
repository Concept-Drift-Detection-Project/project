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
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM, EDDMConfig
from frouros.detectors.concept_drift import PageHinkley, PageHinkleyConfig
from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from river.datasets import synth
from frouros.metrics import PrequentialError
import altair as alt
import time 

def run_simulation():

    # Title and description
    st.title('Concept Drift Detection in a Synthetic Dataset')
    st.write("""
    This app simulates concept drift detection using various methods on a synthetic dataset.
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
        # Dropdown for selecting the drift detection method
        drift_method = st.selectbox(
            "Select Drift Detection Method:",
            ("DDM", "EDDM", "ADWIN", "Page Hinkley")
        )

        # Dropdown for selecting the regression model
        model_choice = st.selectbox(
            "Select the regression model:",
            ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor")
        )

        # Input fields for drift points, step set to 1000
        first_point = st.number_input("Enter the first drift point:", min_value=1000, value=7000, step=1000)
        second_point = st.number_input("Enter the second drift point:", min_value=first_point + 1000, value=11000, step=1000)

        # Parameters based on drift detection method
        if drift_method == "DDM":
            warning_level = st.slider("Warning Level", 0.0, 0.1, 0.01)
            drift_level = st.slider("Drift Level", 1.0, 10.0, 3.0)
            min_num_instances = st.slider("Min Number of Instances", 1, 5000, 1000)
        elif drift_method == "EDDM":
            alpha = st.slider("Alpha", 0.0, 2.0, 0.95)
            beta = st.slider("Beta", 0.0, 2.0, 0.9)
            level = st.slider("Level", 0.0, 10.0, 2.0)
            min_num_instances = st.slider("Min Number of Instances", 1, 100, 30)
        elif drift_method == "ADWIN":
            clock = st.slider("Clock", 1, 256, 128)
            min_window_size = st.slider("Min Window Size", 1, 50, 5)
            min_num_instances = st.slider("Min Number of Instances", 1, 50, 10)
            m_value = st.slider("M value", 1, 20, 9)
        else:  # Page Hinkley
            lambda_ = st.slider("Lambda", 1.0, 100.0, 10.0)
            alpha = st.slider("Alpha", 0.0, 1.0, 0.9999)
            min_num_instances = st.slider("Min Number of Instances", 1, 100, 30)
            delta = st.slider("Delta", 0.0, 0.1, 0.005)

    # Button to trigger computation
    if st.button("Check for Drift"):
        # Start the time measurement
        start_time = time.time()

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

        train = df.iloc[:(first_point // 2)]
        stream = df.iloc[(first_point // 2):]

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
        if drift_method == "DDM":
            config = DDMConfig(
                warning_level=warning_level,
                drift_level=drift_level,
                min_num_instances=min_num_instances,
            )
            detector = DDM(config=config)
        elif drift_method == "EDDM":
            config = EDDMConfig(
                alpha=alpha,
                beta=beta,
                level=level,
                min_num_misclassified_instances=min_num_instances
            )
            detector = EDDM(config=config)
        elif drift_method == "ADWIN":
            config = ADWINConfig(
                clock=clock, 
                min_window_size=min_window_size, 
                min_num_instances=min_num_instances, 
                m=m_value
            )
            detector = ADWIN(config=config)
        else:  # Page Hinkley
            config = PageHinkleyConfig(
                lambda_=lambda_,
                alpha=alpha,
                min_num_instances=min_num_instances,
                delta=delta
            )
            detector = PageHinkley(config=config)

        detected_drifts = []
        false_alarms = 0
        detection_delays = []
        y_preds = []
        errors = []

        metric = PrequentialError(alpha=0.9)

        if drift_method in ['DDM','EDDM']:

            for i in range(len(X_stream)):
                X_i = X_stream[i].reshape(1, -1)
                y_i = y_stream[i].reshape(1, -1)

                # Predict and calculate the error
                y_pred = pipeline.predict(X_i)
                y_preds.append(y_pred[0])
                error = mean_squared_error(y_i, y_pred)
                errors.append(error)

                binary_error = 1 if error > 50.0 else 0  # Adjust threshold based on chosen detector

                # Update the detector with the error
                detector.update(value=binary_error)

                # Check for detected drift
                if detector.drift:
                    detected_drifts.append(i + len(train))  # Adjust index to match the full dataset

                    # Determine if it's a false alarm or not
                    if i + len(train) < odp[0]:
                        false_alarms += 1
                    else:
                        detection_delays.append((i + len(train)) - odp[0])

            false_alarm_rate = (false_alarms / (first_point-len(train))) if detected_drifts else 0
            average_detection_delay = (detection_delays[0]) if detection_delays else None

        else:
            for i in range(len(X_stream)):
                X_i = X_stream[i].reshape(1, -1)
                y_i = y_stream[i].reshape(1, -1)

                # Predict and calculate the error
                y_pred = pipeline.predict(X_i)
                y_preds.append(y_pred[0])
                error = mean_squared_error(y_true=y_i, y_pred=y_pred)
                metric_error = metric(error_value=error)
                errors.append(metric_error)

                # Update the drift detector with the current error
                detector.update(value=error)

                # Check for detected drift
                if detector.drift:
                    detected_drifts.append(i + len(train))

                    # Determine if it's a false alarm or not
                    if i + len(train) < first_point:
                        false_alarms += 1
                    else:
                        detection_delays.append((i + len(train)) - odp[0])
                    detector.reset()

            false_alarm_rate = (false_alarms / (first_point-len(train))) if detected_drifts else 0
            average_detection_delay = int(detection_delays[0]) if detection_delays else None

        # End the time measurement
        elapsed_time = time.time() - start_time

        # Update parameter values with actual results
        with col1:
            st.write(f"False Alarms: {false_alarms}")
            st.write(f"False Alarm Rate: {false_alarm_rate}")
            st.write(f"Average Detection Delay: {average_detection_delay}")
            st.write(f"Time Taken for Simulation: {elapsed_time:.2f} seconds")

        # Prepare the data for Altair visualizations
        error_data = pd.DataFrame({
            'Index': np.arange(len(errors)),
            'Error': errors
        })

        prediction_data = pd.DataFrame({
            'y_true': y_stream.flatten(),
            'y_pred': np.array(y_preds),
            'Index': np.arange(len(y_preds))
        })

        drift_data = pd.DataFrame({
            'Drift Points': detected_drifts,
            'y': [y_preds[idx - len(train)] for idx in detected_drifts]
        })

        # Plot prequential error (Errors over time)
        error_chart = alt.Chart(error_data).mark_line().encode(
            x='Index:Q',
            y='Error:Q'
        ).properties(
            width=800,
            height=300,
            title="Prequential Error Over Time"
        )

        # Plot true vs predicted values
        prediction_chart = alt.Chart(prediction_data).mark_line().encode(
            x='Index:Q',
            y='y_true:Q',
            color=alt.value('red')
        ).properties(
            width=800,
            height=300,
            title="True vs Predicted Values"
        ) + alt.Chart(prediction_data).mark_line().encode(
            x='Index:Q',
            y='y_pred:Q',
            color=alt.value('blue')
        )

        # Plot detected drifts
        drift_chart = alt.Chart(drift_data).mark_circle(color='red', size=80).encode(
            x='Drift Points:Q',
            y='y:Q'
        ).properties(
            title="Detected Drifts"
        )

        # Display charts on the right side
        with col2:
            st.altair_chart(error_chart)
            st.altair_chart(prediction_chart)
            st.altair_chart(drift_chart)


run_simulation() 
