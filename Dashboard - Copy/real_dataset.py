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


def detect():

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

        columns_list = df.columns.tolist()
        y_value = columns_list[-1]
        
        length = len(df)
        st.write("Length = ",length)

        first_point = length // 5
        second_point = length - 1

        # Default values for metrics
        false_alarms = 0
        false_alarm_rate = 0.0
        average_detection_delay = 0.0
        errors1 = []

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

            adp = st.number_input("Enter the assumed drift point:", min_value=1000, value=7000, step=500)


        if st.button("Plot MSE"):

            train = df.iloc[:(first_point // 2)]
            stream = df.iloc[(first_point // 2):]

            X_train = train.drop(columns=y_value).values
            y_train = train[y_value].values

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

            X_stream = stream.drop(columns=y_value).values
            y_stream = stream[y_value].values

            detected_drifts = []
            false_alarms = 0
            detection_delays = []
            y_preds = []

            for i in range(len(X_stream)):
                X_i = X_stream[i].reshape(1, -1)
                y_i = y_stream[i].reshape(1, -1)

                # Predict and calculate the error
                y_pred = pipeline.predict(X_i)
                y_preds.append(y_pred[0])
                error = mean_squared_error(y_i, y_pred)
                errors1.append(error)

            error_data = pd.DataFrame({
                'Index': np.arange(len(errors1)) + (first_point // 2),
                'Mean Squared Error': errors1
            })

            first_10_percent = errors1[:length//10]
            mse1 = np.mean(first_10_percent) if errors1 else 0

            last_10_percent = errors1[-length//10:]
            mse2 = np.mean(last_10_percent) if errors1 else 1.0

            with col1:
                st.write(f"MSE 1 : {mse1} , MSE 2 : {mse2}")

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



        if st.button("Check for Drift"):

            train = df.iloc[:(first_point // 2)]
            stream = df.iloc[(first_point // 2):]

            X_train = train.drop(columns=y_value).values
            y_train = train[y_value].values

            first_10_percent = errors1[:length//10]
            mse1 = np.mean(first_10_percent) if errors1 else 0

            last_10_percent = errors1[(length-length//10):]
            mse2 = np.mean(last_10_percent) if errors1 else 1.0

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

            X_stream = stream.drop(columns=y_value).values
            y_stream = stream[y_value].values

            # # Detector configuration and instantiation
            # if drift_method == "DDM":
            #     config = DDMConfig()
            #     detector = DDM(config=config)
            # elif drift_method == "EDDM":
            #     config = EDDMConfig()
            #     detector = EDDM(config=config)
            # elif drift_method == "ADWIN":
            #     config = ADWINConfig()
            #     detector = ADWIN(config=config)
            # else:  # Page Hinkley
            #     config = PageHinkleyConfig()
            #     detector = PageHinkley(config=config)


            # Detector configuration and instantiation
            if drift_method == "DDM":
                if model_choice == "Linear Regressor":
                    config = DDMConfig(
                        warning_level = 1.65, drift_level = 1.7, min_num_instances = 330
                    )
                    detector = DDM(config=config)
                if model_choice == "SVM Regressor":
                    config = DDMConfig(
                        warning_level = 2.45, drift_level = 2.5, min_num_instances = 90
                    )
                    detector = DDM(config=config)
                if model_choice == "Decision Tree Regressor":
                    config = DDMConfig(
                        warning_level = 2.65, drift_level = 2.7, min_num_instances = 250
                    )
                    detector = DDM(config=config)
                if model_choice == "Random Forest Regressor":
                    config = DDMConfig(
                        warning_level = 5.05, drift_level = 5.1, min_num_instances = 30
                    )
                    detector = DDM(config=config)

            elif drift_method == "EDDM":
                if model_choice == "Linear Regressor":
                    config = EDDMConfig(
                        alpha = 0.90, beta = 0.85, level = 1.95, min_num_misclassified_instances = 50
                    )
                    detector = EDDM(config=config)
                if model_choice == "SVM Regressor":
                    config = EDDMConfig(
                        alpha = 1.0, beta = 0.95, level = 1.0, min_num_misclassified_instances = 110
                    )
                    detector = EDDM(config=config)
                if model_choice == "Decision Tree Regressor":
                    config = EDDMConfig(
                        alpha = 0.90, beta = 0.85, level = 1.55, min_num_misclassified_instances = 170
                    )
                    detector = EDDM(config=config)
                if model_choice == "Random Forest Regressor":
                    config = EDDMConfig(
                        alpha = 0.8, beta = 0.75, level = 1.85, min_num_misclassified_instances = 110
                    )
                    detector = EDDM(config=config)

            elif drift_method == "ADWIN":
                if model_choice == "Linear Regressor":
                    config = ADWINConfig(
                        clock = 1, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10
                    )
                    detector = ADWIN(config=config)
                if model_choice == "SVM Regressor":
                    config = ADWINConfig(
                        clock = 5, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10
                    )
                    detector = ADWIN(config=config)
                if model_choice == "Decision Tree Regressor":
                    config = ADWINConfig(
                        clock = 3, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10 
                    )
                    detector = ADWIN(config=config)
                if model_choice == "Random Forest Regressor":
                    config = ADWINConfig(
                        clock = 3, delta = 0.002, m = 9, min_window_size = 1, min_num_instances = 10
                    )
                    detector = ADWIN(config=config)

            else:  # Page Hinkley
                if model_choice == "Linear Regressor":
                    config = PageHinkleyConfig(
                        delta = 0.005, lambda_ = 14.0, alpha = 0.9999, min_num_instances = 80
                    )
                    detector = PageHinkley(config=config)
                if model_choice == "SVM Regressor":
                    config = PageHinkleyConfig(
                        delta = 0.005, lambda_ = 50.0, alpha = 0.9999, min_num_instances = 30
                    )
                    detector = PageHinkley(config=config)
                if model_choice == "Decision Tree Regressor":
                    config = PageHinkleyConfig(
                        delta = 0.005, lambda_ = 71.0, alpha = 0.9999, min_num_instances = 34
                    )
                    detector = PageHinkley(config=config)
                if model_choice == "Random Forest Regressor":
                    config = PageHinkleyConfig(
                        delta = 0.005, lambda_ = 3.0, alpha = 0.9999, min_num_instances = 10 
                    )
                    detector = PageHinkley(config=config)



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

                binary_error = 1 if error > 0.1*(abs(mse2-mse1)) else 0  # Adjust threshold based on chosen detector

                # Update the detector with the error
                detector.update(value=binary_error)

                # Check for detected drift
                if detector.drift:
                    detected_drifts.append(i + len(train))  # Adjust index to match the full dataset

                    # Determine if it's a false alarm or not
                    if i + len(train) < adp:
                        false_alarms += 1
                    else:
                        detection_delays.append((i + len(train)) - adp)
                else:
                    last_out = i+len(train)

            false_alarm_rate = (false_alarms / (first_point-len(train))) if detected_drifts else 0
            average_detection_delay = last_out+1-adp

            # Update parameter values with actual results
            with col1:
                st.write(f"False Alarms: {false_alarms}")
                st.write(f"False Alarm Rate: {false_alarm_rate}")
                st.write(f"Average Detection Delay: {average_detection_delay}")
                st.write(f"Drift Point: {last_out + 1}") 

            # Prepare the data for Altair visualizations
            error_data = pd.DataFrame({
                'Index': np.arange(len(errors)) + (first_point // 2),
                'Mean Squared Error': errors
            })

            drift_data = pd.DataFrame({
                'Index': np.arange(length),
                'Detected Drift Indicator': [1 if i in detected_drifts else 0 for i in range(length)],
                'Actual Drift Indicator': [1 if i >= adp else 0 for i in range(length)]
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

