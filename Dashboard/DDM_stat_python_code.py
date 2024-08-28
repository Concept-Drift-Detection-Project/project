import river
from river import datasets
from river import linear_model, tree
from river import drift, metrics
from river.datasets import synth
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from frouros.metrics import PrequentialError
from sklearn.pipeline import Pipeline
from frouros.detectors.concept_drift.streaming.window_based.adwin import ADWIN
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.detectors.concept_drift.streaming.statistical_process_control.ddm import DDM
from frouros.detectors.concept_drift.streaming.statistical_process_control.eddm import EDDM,EDDMConfig
from frouros.detectors.concept_drift.streaming.change_detection.page_hinkley import PageHinkley

def initialize_data(dp1, dp2, seed=42):
    """
    Initialize the dataset with specified drift position and seed.
    
    Parameters:
    - a (int): The start position where the drift occurs.
    - b (int): The end position where the drift occurs and also used as the limit.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - pd.DataFrame: DataFrame containing the generated data.
    """
    dataset = synth.FriedmanDrift(
        drift_type='gra',
        position=(dp1, dp2),
        seed=seed
    )
    data = []
    for i, (x, y) in enumerate(dataset):
        x_values = list(x.values())
        data.append(x_values + [y])
        if i >= dp2:  # Limiting to 'b' samples for simplicity
            break
    column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
    df = pd.DataFrame(data, columns=column_names)
    return df

def split_data(df, split):
    """
    Split the dataset into training, validation, and streaming sets.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - split (int): Index in which we split the data in to train and tream.
    
    
    Returns:
    - train (pd.DataFrame): Training set.
    - val (pd.DataFrame): Validation set.
    - stream (pd.DataFrame): Streaming set.
    """
    # Split the DataFrame into train and remaining data
    train = df.iloc[:split]
    stream = df.iloc[split:]
    
    # Split the train data into training and validation sets
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    
    return train, val, stream

def train_model(train,model):
    X_train = train.drop(columns='y').values
    y_train = train['y'].values
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)])
    pipeline.fit(X_train, y_train)
    return pipeline

def calculate_threshold(pipeline, val, quantile_threshold):
    """
    Calculate the quantile-based threshold for error values.
    
    Parameters:
    - pipeline: The trained pipeline used for predictions.
    - val (pd.DataFrame): Validation set containing features and target variable.
    - quantile_threshold (float): The quantile threshold to determine the error threshold.
    
    Returns:
    - threshold (float): The calculated quantile-based threshold.
    """
    # Extract features and target variable from the validation set
    X_val = val.drop(columns='y').values
    y_val = val['y'].values
    
    # Make predictions using the pipeline
    y_pred = pipeline.predict(X_val)
    
    # Calculate the squared errors
    errors_val = (y_val - y_pred) ** 2
    
    # Calculate the quantile-based threshold
    threshold = np.quantile(errors_val, quantile_threshold)
    print(f"Calculated threshold at {quantile_threshold * 100}% quantile: {threshold}")
    
    return threshold

def setup_ddm(warning_level=2.0, drift_level=3.0, min_num_instances=1000):
    """
    Set up the DDM detector with specified configuration parameters.
    
    Parameters:
    - warning_level (float): The warning level for drift detection.
    - drift_level (float): The drift level for drift detection.
    - min_num_instances (int): Minimum number of instances before checking for concept drift.
    
    Returns:
    - detector: An instance of the DDM drift detector.
    """
    config = DDMConfig(
        warning_level=warning_level,
        drift_level=drift_level,
        min_num_instances=min_num_instances
    )
    detector = DDM(config=config)
    return detector

def setup_eddm(alpha=0.95, beta=0.9, level=2.0, min_num_misclassified_instances=30):
    """
    Set up the EDDM (Early Drift Detection Method) detector with specified configuration parameters.
    
    Parameters:
    - alpha (float): Warning zone value.
    - beta (float): Change zone value.
    - level (float): Level factor for drift detection.
    - min_num_misclassified_instances (int): Minimum number of misclassified instances to start detecting drift.
    
    Returns:
    - detector: An instance of the EDDM drift detector.
    """
    config = EDDMConfig(
        alpha=alpha,
        beta=beta,
        level=level,
        min_num_misclassified_instances=min_num_misclassified_instances
    )
    detector = EDDM(config=config)
    return detector

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
    """
    Plot the Mean Squared Error (MSE) values over data points.

    Parameters:
    - errors (List[float]): List of MSE error values to plot.

    Returns:
    - None
    """
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(errors)), errors, label="MSE Error", color='purple')
    plt.xlabel("Instance Index")
    plt.ylabel("Mean Squared Error")
    plt.title("Behavior of Mean Squared Error Over Data Points")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_drift_indicators(data_length: int, detected_drifts: List[int], odp: int):
    """
    Plot drift detection indicators and actual drift indicators.

    Parameters:
    - data_length (int): The length of the data array.
    - detected_drifts (List[int]): List of indices where drifts are detected.
    - odp (int): Index of the initial drift point.
    """
    # Create binary arrays for drift detection and odp
    drift_indicator = np.zeros(data_length)
    for drift_point in detected_drifts:
        if drift_point < data_length:
            drift_indicator[drift_point] = 1

    odp_indicator = np.zeros(data_length)
    for i in range(odp, data_length):
        odp_indicator[i] = 1

    # Plotting the drift indicator and odp indicator
    plt.figure(figsize=(12, 4))
    plt.plot(drift_indicator, label='Detected Drift Indicator', color='red', drawstyle='steps-post')
    plt.plot(odp_indicator, label='Actual Drift Indicator (ODP)', color='green', linestyle='-', drawstyle='steps-post')
    plt.xlabel('Index')
    plt.ylabel('Drift Detection (0 or 1)')
    plt.title('Drift Detection Indicator')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def optimize_quantile(train, val, stream,detector,pipeline,start,end,step_size):
    best_quantile = None
    best_false_alarm_rate = float('inf')
    best_detection_delay = float('inf')
    best_detected_drifts = []  # Initialize to store the best detected drifts

    for quantile in np.arange(start, end, step_size):  # Test from 90th to 99th percentile
        
        threshold = calculate_threshold(pipeline, val, quantile)
        detector = detector  # Not necessary, but kept for clarity
        
        X_stream = stream.drop(columns='y').values
        y_stream = stream['y'].values
        false_alarm_rate, detection_delay, errors, detected_drifts = process_stream(X_stream, y_stream, pipeline, detector, threshold, odp=len(train))

        print(f"Quantile: {quantile:.2f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
        print(f"Detection Delay: {detection_delay}")
        #print(f"Detected Drifts: {detected_drifts}\n")

        if false_alarm_rate < best_false_alarm_rate or (false_alarm_rate == best_false_alarm_rate and (detection_delay is None or detection_delay < best_detection_delay)):
            best_quantile = quantile
            best_false_alarm_rate = false_alarm_rate
            best_detection_delay = detection_delay
            best_detected_drifts = detected_drifts  # Update the best detected drifts

    print(f"Best Quantile: {best_quantile}")
    print(f"Best False Alarm Rate: {best_false_alarm_rate}")
    print(f"Best Detection Delay: {best_detection_delay}")
    print(f"Best Detected Drifts: {best_detected_drifts}")
    
    return best_quantile, best_false_alarm_rate, best_detection_delay, best_detected_drifts

def optimize_drift_level(train, val, stream, pipeline, quantile_threshold,start,end,step_size):
    best_drift_level = None
    best_false_alarm_rate = float('inf')
    best_detection_delay = float('inf')
    best_detected_drifts = []

    # Fixed quantile threshold to calculate the threshold
    threshold = calculate_threshold(pipeline, val, quantile_threshold)
    
    # Test various drift levels
    for drift_level in np.arange(start,end,step_size):  # Adjust drift level range and step as needed
        detector = setup_ddm(drift_level-0.01,drift_level,30)
        
        X_stream = stream.drop(columns='y').values
        y_stream = stream['y'].values
        false_alarm_rate, detection_delay, errors, detected_drifts = process_stream(
            X_stream, y_stream, pipeline, detector, threshold, odp=len(train))

        print(f"Drift Level: {drift_level:.2f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
        print(f"Detection Delay: {detection_delay}")
        #print(f"Detected Drifts: {detected_drifts}\n")

        if false_alarm_rate < best_false_alarm_rate or (false_alarm_rate == best_false_alarm_rate and (detection_delay is None or detection_delay < best_detection_delay)):
            best_drift_level = drift_level
            best_false_alarm_rate = false_alarm_rate
            best_detection_delay = detection_delay
            best_detected_drifts = detected_drifts

    print(f"Best Drift Level: {best_drift_level}")
    print(f"Best False Alarm Rate: {best_false_alarm_rate}")
    print(f"Best Detection Delay: {best_detection_delay}")
    print(f"Best Detected Drifts: {best_detected_drifts}")
    
    return best_drift_level, best_false_alarm_rate, best_detection_delay, best_detected_drifts

def optimize_drift_level(train, val, stream, pipeline, quantile_threshold,start,end,step_size,level=0.2,mm=30):
    best_drift_level = None
    best_false_alarm_rate = float('inf')
    best_detection_delay = float('inf')
    best_detected_drifts = []

    # Fixed quantile threshold to calculate the threshold
    threshold = calculate_threshold(pipeline, val, quantile_threshold)
    
    # Test various drift levels
    for drift_level in np.arange(start,end,step_size):  # Adjust drift level range and step as needed
        detector = setup_eddm(drift_level,drift_level-0.01,level,mm)
        
        X_stream = stream.drop(columns='y').values
        y_stream = stream['y'].values
        false_alarm_rate, detection_delay, errors, detected_drifts = process_stream(
            X_stream, y_stream, pipeline, detector, threshold, odp=len(train))

        print(f"Drift Level: {drift_level:.2f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
        print(f"Detection Delay: {detection_delay}")
        #print(f"Detected Drifts: {detected_drifts}\n")

        if detection_delay is None or detection_delay < best_detection_delay or (detection_delay == best_detection_delay and false_alarm_rate < best_false_alarm_rate ):
            best_drift_level = drift_level
            best_false_alarm_rate = false_alarm_rate
            best_detection_delay = detection_delay
            best_detected_drifts = detected_drifts

    print(f"Best Drift Level: {best_drift_level}")
    print(f"Best False Alarm Rate: {best_false_alarm_rate}")
    print(f"Best Detection Delay: {best_detection_delay}")
    print(f"Best Detected Drifts: {best_detected_drifts}")
    
    return best_drift_level, best_false_alarm_rate, best_detection_delay, best_detected_drifts

def optimize_level(train, val, stream, pipeline, quantile_threshold,drift_level,start,end,step_size,mm=30):
    best_drift_level = None
    best_false_alarm_rate = float('inf')
    best_detection_delay = float('inf')
    best_detected_drifts = []

    # Fixed quantile threshold to calculate the threshold
    threshold = calculate_threshold(pipeline, val, quantile_threshold)
    
    # Test various drift levels
    for level in np.arange(start,end,step_size):  # Adjust drift level range and step as needed
        detector = setup_eddm(drift_level,drift_level-0.01,level,mm)
        
        X_stream = stream.drop(columns='y').values
        y_stream = stream['y'].values
        false_alarm_rate, detection_delay, errors, detected_drifts = process_stream(
            X_stream, y_stream, pipeline, detector, threshold, odp=len(train))

        print(f"Drift Level: {drift_level:.2f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
        print(f"Detection Delay: {detection_delay}")
        #print(f"Detected Drifts: {detected_drifts}\n")

        if detection_delay is None or detection_delay < best_detection_delay or (detection_delay == best_detection_delay and false_alarm_rate < best_false_alarm_rate ):
            best_drift_level = drift_level
            best_false_alarm_rate = false_alarm_rate
            best_detection_delay = detection_delay
            best_detected_drifts = detected_drifts

    print(f"Best Drift Level: {best_drift_level}")
    print(f"Best False Alarm Rate: {best_false_alarm_rate}")
    print(f"Best Detection Delay: {best_detection_delay}")
    print(f"Best Detected Drifts: {best_detected_drifts}")
    
    return best_drift_level, best_false_alarm_rate, best_detection_delay, best_detected_drifts

def optimize_level(train, val, stream, pipeline, quantile_threshold,drift_level,start,end,step_size,level):
    best_drift_level = None
    best_false_alarm_rate = float('inf')
    best_detection_delay = float('inf')
    best_detected_drifts = []

    # Fixed quantile threshold to calculate the threshold
    threshold = calculate_threshold(pipeline, val, quantile_threshold)
    
    # Test various drift levels
    for mm in np.arange(start,end,step_size):  # Adjust drift level range and step as needed
        detector = setup_eddm(drift_level,drift_level-0.01,level,mm)
        
        X_stream = stream.drop(columns='y').values
        y_stream = stream['y'].values
        false_alarm_rate, detection_delay, errors, detected_drifts = process_stream(
            X_stream, y_stream, pipeline, detector, threshold, odp=len(train))

        print(f"Drift Level: {drift_level:.2f}")
        print(f"Threshold: {threshold:.2f}")
        print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
        print(f"Detection Delay: {detection_delay}")
        #print(f"Detected Drifts: {detected_drifts}\n")

        if detection_delay is None or detection_delay < best_detection_delay or (detection_delay == best_detection_delay and false_alarm_rate < best_false_alarm_rate ):
            best_drift_level = drift_level
            best_false_alarm_rate = false_alarm_rate
            best_detection_delay = detection_delay
            best_detected_drifts = detected_drifts

    print(f"Best Drift Level: {best_drift_level}")
    print(f"Best False Alarm Rate: {best_false_alarm_rate}")
    print(f"Best Detection Delay: {best_detection_delay}")
    print(f"Best Detected Drifts: {best_detected_drifts}")
    
    return best_drift_level, best_false_alarm_rate, best_detection_delay, best_detected_drifts


a = 2000
b = 3000
df = initialize_data(a, b,42)

split_index = 1000

train, val, stream = split_data(df, split_index)

model1=LinearRegression()
model2=DecisionTreeRegressor()
model3=RandomForestRegressor()
model4=SVR()

model = model1

pipeline=train_model(train,model)
quantile_threshold = 0.95 
threshold=calculate_threshold(pipeline, val, quantile_threshold)
detector=setup_ddm(warning_level=2.0, drift_level=3.0, min_num_instances=30)
detector=setup_eddm(alpha=0.95, beta=0.9, level=2.0, min_num_misclassified_instances=30)

odp=a

X = stream.drop(columns='y').values
y = stream['y'].values

false_alarm_rate, detection_delay, errors, detected_drifts= process_stream(X, y, pipeline, detector, threshold, odp)

print(f"False Alarm Rate: {false_alarm_rate:.2f}%")
print(f"Detection Delay: {detection_delay}")
print(f"Detected Drifts: {detected_drifts}\n")

plot_errors(errors)

plot_drift_indicators(b, detected_drifts, odp)

pipeline=train_model(train,model)

quantile_threshold = 0.95 

detector1 = setup_ddm(warning_level=2.0, drift_level=3.0, min_num_instances=30)
detector2 = setup_eddm(alpha=0.95, beta=0.9, level=2.0, min_num_misclassified_instances=30)

detector = detector1

start=0.5
end=0.95
step_size=0.05

best_quantile, best_false_alarm_rate, best_detection_delay, best_detected_drifts = optimize_quantile(train, val, stream,detector,pipeline,start,end,step_size)

plot_drift_indicators(b, best_detected_drifts, odp)

start=0.05
end=0.5
step_size=0.05

best_drift_level, best_false_alarm_rate, best_detection_delay, best_detected_drifts=optimize_drift_level(train, val, stream, pipeline, best_quantile,start,end,step_size)

plot_drift_indicators(b, best_detected_drifts, odp)

