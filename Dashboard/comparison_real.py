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
import altair as alt


def compare():
    
    # Title and description
    st.title('Concept Drift Detection on User-Uploaded Dataset')
    st.write("""
    This app simulates concept drift detection using various methods on a user-uploaded dataset.
    It displays the performance metrics and drift indicators for all detectors.
    """)

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        # Read dataset
        df = pd.read_csv(uploaded_file)
        
        # Assuming the target variable (y) is the last column
        st.write(f"Dataset Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        
        # Allow the user to select the target variable (y)
        target_col = st.selectbox("Select the target variable (y):", df.columns)
        
        # All columns except the selected target variable will be used as features
        feature_cols = [col for col in df.columns if col != target_col]

        # Split the data into train and stream
        first_point = st.number_input("Enter the first drift point:", min_value=1000, value=min(7000, len(df)//2), step=1000)
        second_point = st.number_input("Enter the second drift point:", min_value=first_point + 1000, value=min(11000, len(df)), step=1000)

        if st.button("Run Drift Detection"):
            train = df.iloc[:(first_point // 2)]
            stream = df.iloc[(first_point // 2):]

            X_train = train[feature_cols].values
            y_train = train[target_col].values

            # Choose regression model
            model_choice = st.selectbox(
                "Select the regression model:",
                ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor")
            )

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

            odp = [first_point]  # Actual drift point

            X_stream = stream[feature_cols].values
            y_stream = stream[target_col].values

            # List of detectors
            detectors = [
                ('DDM', DDM(config=DDMConfig())),
                ('EDDM', EDDM(config=EDDMConfig())),
                ('ADWIN', ADWIN(config=ADWINConfig())),
                ('Page Hinkley', PageHinkley(config=PageHinkleyConfig()))
            ]

            # Dictionary to store metrics for each detector
            drift_results = {}
            error_data = {}
            drift_data = {}

            for detector_name, detector in detectors:
                detected_drifts = []
                false_alarms = 0
                detection_delays = []
                y_preds = []
                errors = []

                for i in range(len(X_stream)):
                    X_i = X_stream[i].reshape(1, -1)
                    y_i = y_stream[i].reshape(1, -1)

                    # Predict and calculate the error
                    y_pred = pipeline.predict(X_i)
                    y_preds.append(y_pred[0])
                    error = mean_squared_error(y_i, y_pred)
                    errors.append(error)

                    binary_error = 1 if error > 50.0 else 0

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

                false_alarm_rate = false_alarms / len(detected_drifts) if detected_drifts else 0
                average_detection_delay = (detection_delays[0]) if detection_delays else None

                # Save results
                drift_results[detector_name] = {
                    'False Alarms': false_alarms,
                    'False Alarm Rate': false_alarm_rate,
                    'Average Detection Delay': average_detection_delay
                }

                # Prepare data for visualizations
                error_data[detector_name] = pd.DataFrame({
                    'Index': np.arange(len(errors)) + (first_point // 2),
                    'Mean Squared Error': errors
                })

                drift_data[detector_name] = pd.DataFrame({
                    'Index': np.arange(len(df)),
                    'Detected Drift Indicator': [1 if i in detected_drifts else 0 for i in range(len(df))],
                    'Actual Drift Indicator': [1 if i >= odp[0] else 0 for i in range(len(df))]
                })

            # Display results in a table
            results_df = pd.DataFrame(drift_results).T
            st.write(results_df)

            # Plot False Alarm Rate
            false_alarm_chart = alt.Chart(results_df.reset_index()).mark_bar().encode(
                x=alt.X('index:N', title='Detector'),
                y=alt.Y('False Alarm Rate:Q', title='False Alarm Rate')
            ).properties(
                width=600,
                height=300,
                title='False Alarm Rate for Each Detector'
            )
            st.altair_chart(false_alarm_chart, use_container_width=True)

            # Plot Detection Delay
            detection_delay_chart = alt.Chart(results_df.reset_index()).mark_bar().encode(
                x=alt.X('index:N', title='Detector'),
                y=alt.Y('Average Detection Delay:Q', title='Detection Delay')
            ).properties(
                width=600,
                height=300,
                title='Average Detection Delay for Each Detector'
            )
            st.altair_chart(detection_delay_chart, use_container_width=True)

            # Right side (graphs)
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

