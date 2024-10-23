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
from frouros.metrics import PrequentialError
from river.datasets import synth
import altair as alt


def compare():
    
    # Title and description
    st.title('Concept Drift Detection in a Synthetic Dataset')
    st.write("""
    This app simulates concept drift detection using various methods on a synthetic dataset.
    It displays the performance metrics and drift indicators for all detectors.
    """)

    # Left and Right side columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Drift Detection Results Table")
        model_choice = st.selectbox(
            "Select the regression model:",
            ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor")
        )

        # Input fields for drift points
        first_point = st.number_input("Enter the first drift point:", min_value=1000, value=7000, step=1000)
        second_point = st.number_input("Enter the second drift point:", min_value=first_point + 1000, value=first_point+4000, step=1000)

        # Button to trigger computation
        if st.button("Run Drift Detection"):
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
                if i >= second_point:
                    break

            # Create DataFrame
            column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
            df = pd.DataFrame(data, columns=column_names)

            train = df.iloc[:(first_point // 2)]
            stream = df.iloc[(first_point // 2):]

            X_train = train.drop(columns='y').values
            y_train = train['y'].values

            # # List of detectors
            # detectors = [
            #     ('DDM', DDM(config=DDMConfig())),
            #     ('EDDM', EDDM(config=EDDMConfig())),
            #     ('ADWIN', ADWIN(config=ADWINConfig())),
            #     ('Page Hinkley', PageHinkley(config=PageHinkleyConfig()))
            # ] 


            # Select the regression model based on the dropdown choice
            if model_choice == "Linear Regressor":
                model = LinearRegression()
                detectors = [
                    ('DDM', DDM(config=DDMConfig(
                        warning_level = 1.65, drift_level = 1.7, min_num_instances = 330
                    ))),
                    ('EDDM', EDDM(config=EDDMConfig(
                        alpha = 0.90, beta = 0.85, level = 1.95, min_num_misclassified_instances = 50
                    ))),
                    ('ADWIN', ADWIN(config=ADWINConfig(
                        clock = 1, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10
                    ))),
                    ('Page Hinkley', PageHinkley(config=PageHinkleyConfig(
                        delta = 0.005, lambda_ = 14.0, alpha = 0.9999, min_num_instances = 80
                    )))
                ]
            elif model_choice == "SVM Regressor":
                model = SVR()
                detectors = [
                    ('DDM', DDM(config=DDMConfig(
                        warning_level = 2.45, drift_level = 2.5, min_num_instances = 90
                    ))),
                    ('EDDM', EDDM(config=EDDMConfig(
                        alpha = 1.0, beta = 0.95, level = 1.0, min_num_misclassified_instances = 110
                    ))),
                    ('ADWIN', ADWIN(config=ADWINConfig(
                        clock = 5, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10
                    ))),
                    ('Page Hinkley', PageHinkley(config=PageHinkleyConfig(
                        delta = 0.005, lambda_ = 50.0, alpha = 0.9999, min_num_instances = 30
                    )))
                ]
            elif model_choice == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
                detectors = [
                    ('DDM', DDM(config=DDMConfig(
                        warning_level = 2.65, drift_level = 2.7, min_num_instances = 250
                    ))),
                    ('EDDM', EDDM(config=EDDMConfig(
                        alpha = 0.90, beta = 0.85, level = 1.55, min_num_misclassified_instances = 170
                    ))),
                    ('ADWIN', ADWIN(config=ADWINConfig(
                        clock = 3, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10 
                    ))),
                    ('Page Hinkley', PageHinkley(config=PageHinkleyConfig(
                        delta = 0.005, lambda_ = 71.0, alpha = 0.9999, min_num_instances = 34
                    )))
                ]
            else:
                model = RandomForestRegressor()
                detectors = [
                    ('DDM', DDM(config=DDMConfig(
                        warning_level = 5.05, drift_level = 5.1, min_num_instances = 30
                    ))),
                    ('EDDM', EDDM(config=EDDMConfig(
                        alpha = 0.8, beta = 0.75, level = 1.85, min_num_misclassified_instances = 110
                    ))),
                    ('ADWIN', ADWIN(config=ADWINConfig(
                        clock = 3, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10
                    ))),
                    ('Page Hinkley', PageHinkley(config=PageHinkleyConfig(
                        delta = 0.005, lambda_ = 3.0, alpha = 0.9999, min_num_instances = 10 
                    )))
                ]

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
            pipeline.fit(X=X_train, y=y_train)

            odp = [first_point]  # Actual drift point

            X_stream = stream.drop(columns='y').values
            y_stream = stream['y'].values

            
            # Dictionary to store metrics for each detector
            drift_results = {}
            error_data = {}
            drift_data = {}

            metric = PrequentialError(alpha=0.9)

            for detector_name, detector in detectors:
                detected_drifts = []
                false_alarms = 0
                detection_delays = []
                y_preds = []
                errors = []

                if detector_name in ['DDM','EDDM']:
                    for i in range(len(X_stream)):
                        X_i = X_stream[i].reshape(1, -1)
                        y_i = y_stream[i].reshape(1, -1)

                        threshold = 50.0 # if detector_name=='DDM' else 83.146

                        # Predict and calculate the error
                        y_pred = pipeline.predict(X_i)
                        y_preds.append(y_pred[0])
                        error = mean_squared_error(y_i, y_pred)
                        errors.append(error)

                        binary_error = 1 if error > threshold else 0

                        # Update the detector with the error
                        detector.update(value=binary_error)

                        # Check for detected drift
                        if detector.drift:
                            detected_drifts.append(i + len(train))

                            # False alarm detection
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
                        # error = metric(error_value=mean_squared_error(y_true=y_i, y_pred=y_pred))
                        errors.append(metric_error)

                        # Update Page Hinkley with the error (0 for correct, 1 for incorrect)
                        # detector.update(value=1 if error > 0 else 0)

                        # Step 4: Update the drift detector with the current error
                        detector.update(value=error)

                        # Step 5: Check for detected drift
                        if detector.drift:
                            # print(f"Change detected at step {i}")
                            detected_drifts.append(i + len(train))

                            # Determine if it's a false alarm or not
                            if i + len(train) < first_point:  # Compare with the original drift point
                                false_alarms += 1
                            else:
                                #if isFirst:
                                    #isFirst = False
                                detection_delays.append((i + len(train)) - odp[0])

                            detector.reset()

                        # Check for detected warning
                        #if not warning_flag and detector.warning:
                        #   print(f"Warning detected at step {i}")
                        #  warning_flag = True

                    false_alarm_rate = (false_alarms / (first_point-len(train))) if detected_drifts else 0
                    average_detection_delay = int(detection_delays[0]) if detection_delays else None


                # Save results
                drift_results[detector_name] = {
                    'False Alarms': false_alarms,
                    'False Alarm Rate': false_alarm_rate,
                    'Detection Delay': average_detection_delay,
                    'Average Measure': (false_alarms + average_detection_delay) / 2 
                }

                # Prepare data for visualizations
                error_data[detector_name] = pd.DataFrame({
                    'Index': np.arange(len(errors)) + (first_point // 2),
                    'Mean Squared Error': errors
                })

                detected_stream = [0]*len(data)

                for p in detected_drifts:
                    detected_stream[p] = 1 
                    if p > odp[0] :
                        detected_stream[p:] = [1]*(len(data) - p) 
                        break 

                drift_data[detector_name] = pd.DataFrame({
                    'Index': np.arange(len(data)),
                    'Detected Drift Indicator': detected_stream, 
                    'Actual Drift Indicator': [1 if i >= odp[0] else 0 for i in range(len(data))]
                })

            with col1:
                # Display results in a table
                results_df = pd.DataFrame(drift_results).T
                st.write(results_df)

                # Plot False Alarm Rate
                false_alarm_chart = alt.Chart(results_df.reset_index()).mark_bar().encode(
                    x=alt.X('index:N', title='Detector', sort=None),
                    y=alt.Y('False Alarm Rate:Q', title='False Alarm Rate')
                ).properties(
                    width=600,
                    height=300,
                    title='False Alarm Rate for Each Detector'
                )
                st.altair_chart(false_alarm_chart, use_container_width=True)

                # Plot Detection Delay
                detection_delay_chart = alt.Chart(results_df.reset_index()).mark_bar().encode(
                    x=alt.X('index:N', title='Detector', sort=None),
                    y=alt.Y('Detection Delay:Q', title='Detection Delay')
                ).properties(
                    width=600,
                    height=300,
                    title='Detection Delay for Each Detector'
                )
                st.altair_chart(detection_delay_chart, use_container_width=True)

                # Plot Average Measure
                detection_delay_chart = alt.Chart(results_df.reset_index()).mark_bar().encode(
                    x=alt.X('index:N', title='Detector', sort=None),
                    y=alt.Y('Average Measure:Q', title='Average Measure')
                ).properties(
                    width=600,
                    height=300,
                    title='Average Measure for Each Detector'
                )
                st.altair_chart(detection_delay_chart, use_container_width=True)
                

            # Right side (graphs)
            with col2:
                st.subheader('Drift Detection Indicator')

                # Generate charts for each detector
                for detector_name in detectors:
                    detector_name = detector_name[0]

                    # Drift Detection chart
                    drift_chart = alt.Chart(drift_data[detector_name]).mark_line().transform_fold(
                        ['Detected Drift Indicator', 'Actual Drift Indicator'],
                        as_=['Indicator Type', 'Value']
                    ).encode(
                        x='Index',
                        y=alt.Y('Value:Q', axis=alt.Axis(title='Drift Detection (0 or 1)')),
                        color='Indicator Type:N'
                    ).properties(
                        width=600,
                        height=300,
                        title=f"Drift Detection Indicator - {detector_name}"
                    )
                    st.altair_chart(drift_chart, use_container_width=True)

