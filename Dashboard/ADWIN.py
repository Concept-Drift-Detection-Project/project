import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import altair as alt
from river.datasets import synth
from frouros.detectors.concept_drift import ADWIN, ADWINConfig
from frouros.metrics import PrequentialError

# Initialize Streamlit app
st.title("Concept Drift Detection Dashboard")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select a Model", 
    ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Regression"]
)

# Sidebar for ADWIN configuration
st.sidebar.title("ADWIN Configuration")
clock = st.sidebar.slider("Clock", 1, 256, 128)
min_window_size = st.sidebar.slider("Min Window Size", 1, 50, 5)
min_num_instances = st.sidebar.slider("Min Number of Instances", 1, 50, 10)
m_value = st.sidebar.slider("M value", 1, 20, 9)

# Input fields for drift points, step set to 1000
first_point = st.number_input("Enter the first drift point:", min_value=1000, value=7000, step=1000)
second_point = st.number_input("Enter the second drift point:", min_value=first_point+1000, value=11000, step=1000)

# Generate synthetic dataset with concept drift
dataset = synth.FriedmanDrift(drift_type='gra', position=(first_point, second_point), seed=42)

# Create the data container and DataFrame
data = []
for i, (x, y) in enumerate(dataset):
    x_values = list(x.values())
    data.append(x_values + [y])
    if i >= second_point:
        break

column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
df = pd.DataFrame(data, columns=column_names)

# Split the data into training and streaming sets
train = df.iloc[:(first_point//2)]
stream = df.iloc[(first_point//2):]

X_train = train.drop(columns='y').values
y_train = train['y'].values
X_test = stream.drop(columns='y').values
y_test = stream['y'].values

# Model initialization based on user selection
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor()
elif model_choice == "Decision Tree":
    model = DecisionTreeRegressor()
else:
    model = SVR()

# Pipeline creation and fitting
pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
pipeline.fit(X=X_train, y=y_train)

# ADWIN Detector initialization
config = ADWINConfig(clock=clock, min_window_size=min_window_size, min_num_instances=min_num_instances, m=m_value)
detector = ADWIN(config=config)

detected_drifts = []
false_alarms = 0
detection_delays = []
y_preds = []
errors = []

isFirst = True
drift_position = 1000

# Prequential Error Metric
metric = PrequentialError(alpha=0.9)

# Concept drift detection loop
for i in range(len(X_test)):
    X_i = X_test[i].reshape(1, -1)
    y_i = y_test[i].reshape(1, -1)

    y_pred = pipeline.predict(X_i)
    y_preds.append(y_pred[0])
    error = mean_squared_error(y_true=y_i, y_pred=y_pred)
    errors.append(error)

    # Update ADWIN with the error
    detector.update(value=error)

    if detector.drift:
        detected_drifts.append(i)
        if i < drift_position:
            false_alarms += 1
        else:
            if isFirst:
                isFirst = False
                detection_delays.append(i - drift_position)

# Calculate metrics
false_alarm_rate = false_alarms / len(detected_drifts) if detected_drifts else 0
average_detection_delay = np.mean(detection_delays) if detection_delays else None

# Display metrics on the sidebar
st.sidebar.markdown("## Metrics")
st.sidebar.write(f"False Alarms: {false_alarms}")
st.sidebar.write(f"False Alarm Rate: {false_alarm_rate:.4f}")
st.sidebar.write(f"Average Detection Delay: {average_detection_delay}")

# Prepare data for visualization
visualization_data = pd.DataFrame({
    "Index": range(len(y_test)),
    "Actual y": y_test,
    "Predicted y": y_preds,
    "Error": errors
})

drift_indicator = np.zeros(len(X_test))
for drift_point in detected_drifts:
    drift_indicator[drift_point] = 1

visualization_data["Drift Indicator"] = drift_indicator

# Line Chart: Actual vs Predicted y values
st.subheader("Line Chart: Actual vs Predicted y values")
line_chart = alt.Chart(visualization_data).mark_line().encode(
    x='Index',
    y=alt.Y('Actual y', title='y values'),
    color=alt.value('blue')
).properties(
    title='Actual vs Predicted y values'
) + alt.Chart(visualization_data).mark_line(strokeDash=[5,5]).encode(
    x='Index',
    y=alt.Y('Predicted y', title='y values'),
    color=alt.value('red')
)

st.altair_chart(line_chart, use_container_width=True)

# Behavior of Mean Squared Error Over Data Points
st.subheader("Behavior of Mean Squared Error Over Data Points")
error_chart = alt.Chart(visualization_data).mark_line().encode(
    x='Index',
    y=alt.Y('Error', title='Mean Squared Error'),
    color=alt.value('purple')
).properties(
    title='Behavior of Mean Squared Error Over Data Points'
)

st.altair_chart(error_chart, use_container_width=True)

# Drift Indicator Visualization
st.subheader("Drift Indicator Visualization")
drift_data = pd.DataFrame({
    "Index": range(len(X_test)),
    "Drift Indicator": drift_indicator,
    "Initial Drift": [1 if i >= drift_position else 0 for i in range(len(X_test))]
})

drift_chart = alt.Chart(drift_data).mark_area(opacity=0.3).encode(
    x='Index',
    y=alt.Y('Drift Indicator', title='Drift Indicator Value'),
    color=alt.value('red')
).properties(
    title='Drift Indicator Visualization'
) + alt.Chart(drift_data).mark_area(opacity=0.3).encode(
    x='Index',
    y=alt.Y('Initial Drift', title='Drift Indicator Value'),
    color=alt.value('blue')
)

st.altair_chart(drift_chart, use_container_width=True)
