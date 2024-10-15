import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from river.datasets import synth
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM

# App Title
st.set_page_config(layout="wide", page_title="Concept Drift Detection Using DDM in a Synthetic Dataset")

st.markdown(
    """
    <h1 style='text-align: center; color: white;'>Concept Drift Detection Using DDM in a Synthetic Dataset</h1>
    <p style='text-align: center; color: white;'>This app simulates concept drift detection using the DDM method on a synthetic dataset. It also displays the performance metrics and drift indicators.</p>
    """,
    unsafe_allow_html=True,
)

# Sidebar - Input Parameters
with st.sidebar:
    st.header("Parameter Values")
    st.write("Drift Detector: DDM")
    regressor_model = st.selectbox("Select the regression model:", 
                                   ("Linear Regressor", "Random Forest Regressor", "Decision Tree Regressor", "SVM Regressor"))
    first_point = st.number_input("Enter the first drift point:", value=5000, min_value=1)
    second_point = st.number_input("Enter the second drift point:", value=10000, min_value=1)

# Load the dataset
dataset = synth.FriedmanDrift(drift_type='gra', position=(first_point, second_point), seed=42)

# Initialize the data containers
data = []
for i, (x, y) in enumerate(dataset):
    x_values = list(x.values())
    data.append(x_values + [y])
    if i >= second_point:  # Limiting to 5000 samples for simplicity
        break

# Define the column names
column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']

# Create the DataFrame
df = pd.DataFrame(data, columns=column_names)

# Split the data
train = df.iloc[:(first_point//2)]
stream = df.iloc[(first_point//2):]

# Define the models
if regressor_model == "Linear Regressor":
    model = LinearRegression()
elif regressor_model == "Random Forest Regressor":
    model = RandomForestRegressor()
elif regressor_model == "Decision Tree Regressor":
    model = DecisionTreeRegressor()
else:
    model = SVR()

# Pipeline for scaling and model fitting
pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

# Fit the pipeline on the training data
pipeline.fit(train.drop(columns='y'), train['y'])

X = stream.drop(columns='y').values
y = stream['y'].values

# Instantiate EDDM detector
detector = EDDM()
detected_drifts = []
false_alarms = 0
detection_delays = []

y_preds = []
errors = []
odp = [first_point]  # Actual drift point
warning_flag = False

for i in range(len(X)):
    X_i = X[i].reshape(1, -1)
    y_i = y[i].reshape(1, -1)
    
    # Predict and calculate the error
    y_pred = pipeline.predict(X_i)
    y_preds.append(y_pred[0])
    error = mean_squared_error(y_i, y_pred)
    errors.append(error)
    
    binary_error = 1 if error > 83.146 else 0
    
    # Update EDDM with the error
    detector.update(value=binary_error)
    
    # Check for detected drift
    if detector.drift:
        detected_drifts.append(i + len(train))
        if i + len(train) < odp[0]:
            false_alarms += 1
        else:
            detection_delays.append((i + len(train)) - odp[0])

false_alarm_rate = false_alarms / len(detected_drifts) if detected_drifts else 0
detection_delay = (detection_delays[0]) if detection_delays else None

# Display metrics
st.write(f"False Alarms: {false_alarms}")
st.write(f"False Alarm Rate: {false_alarm_rate}")
st.write(f"Average Detection Delay: {detection_delay}")

# Plotting Mean Squared Error
mse_df = pd.DataFrame({"Index": range(len(errors)), "Mean Squared Error": errors})
mse_chart = alt.Chart(mse_df).mark_line(color="purple").encode(
    x=alt.X('Index', title='Index'),
    y=alt.Y('Mean Squared Error', title='Mean Squared Error')
).properties(
    title="Mean Squared Error Over Data Points",
    width=700,
    height=300
)

# Create binary arrays for drift detection and odp
drift_indicator = np.zeros(len(data))
for drift_point in detected_drifts:
    drift_indicator[drift_point] = 1

odp_indicator = np.zeros(len(data))
for i in range(odp[0], len(data)):
    odp_indicator[i] = 1

# Plotting Drift Detection Indicator
drift_df = pd.DataFrame({
    "Index": range(len(data)),
    "Actual Drift Indicator": odp_indicator,
    "Detected Drift Indicator": drift_indicator
})

drift_chart = alt.Chart(drift_df).transform_fold(
    ["Actual Drift Indicator", "Detected Drift Indicator"],
    as_=['Indicator Type', 'Drift Detection']
).mark_line().encode(
    x='Index',
    y='Drift Detection:Q',
    color='Indicator Type:N'
).properties(
    title="Drift Detection Indicator",
    width=700,
    height=300
)

# Display the plots side by side
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(mse_chart, use_container_width=True)
with col2:
    st.altair_chart(drift_chart, use_container_width=True)
