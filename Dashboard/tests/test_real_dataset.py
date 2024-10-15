import sys
import os 

# Add the project root directory to Python's path, so the test can find home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import real_dataset 

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


@patch('default.st.altair_chart')
@patch('default.st.number_input')
@patch('default.st.selectbox')
@patch('default.st.button')
@patch('default.st.file_uploader')
@patch('default.st.write')
@patch('default.st.title')
@patch('default.st.columns')
@patch('default.pd.read_csv')  # Mock pandas read_csv to simulate file reading
def test_detect(mock_read_csv, mock_columns, mock_title, mock_write, mock_file_uploader, 
                mock_button, mock_selectbox, mock_number_input, mock_altair_chart):
    
    # Prepare a mock dataset
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(10000),
        'feature2': np.random.randn(10000),
        'target': np.random.randn(10000)
    })
    
    # Mock the file uploader to simulate a file being uploaded
    mock_file_uploader.return_value = MagicMock()
    mock_read_csv.return_value = sample_data  # Return the sample dataset when pd.read_csv is called
    
    # Mock the values for the selectbox and number inputs
    mock_selectbox.side_effect = ["DDM", "Linear Regressor"]  # Simulate drift method and regression model choices
    mock_number_input.return_value = 7000  # Simulate the assumed drift point

    # Mock columns
    mock_col1 = MagicMock()
    mock_col2 = MagicMock()
    mock_columns.return_value = (mock_col1, mock_col2)

    # Simulate button presses
    mock_button.side_effect = [False, True]  # First "Plot MSE", then "Check for Drift"

    # Run the detect function 
    real_dataset.detect()

    # Check that Streamlit components were called correctly
    mock_title.assert_called_once_with('Concept Drift Detection on User-Uploaded Dataset')
    mock_file_uploader.assert_called_once_with("Upload your dataset (CSV format)", type="csv")
    
    # Check the number input and selectboxes
    mock_selectbox.assert_any_call("Select Drift Detection Method:", ("DDM", "EDDM", "ADWIN", "Page Hinkley"))
    mock_selectbox.assert_any_call("Select the regression model:", 
                                   ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor"))
    mock_number_input.assert_called_once_with("Enter the assumed drift point:", min_value=1000, value=7000, step=500)

    # Ensure that buttons are clicked in the correct order
    assert mock_button.call_count == 2

    # Verify MSE calculation and charting (when "Plot MSE" is clicked)
    # Check that the MSE chart is rendered correctly
    mock_altair_chart.assert_any_call()  # Check that MSE chart is rendered once
    
    # Verify that st.write is called with the MSE values
    mock_write.assert_any_call(f"MSE 1 : {np.mean(sample_data['target'][:1000])} , MSE 2 : {np.mean(sample_data['target'][-1000:])}")

    # After the "Check for Drift" button is pressed
    # Verify that drift detection results are displayed correctly
    mock_write.assert_any_call("False Alarms: 0")
    mock_write.assert_any_call("False Alarm Rate: 0")
    mock_write.assert_any_call(f"Average Detection Delay: {len(sample_data)-7000+1}")
    mock_write.assert_any_call(f"Drift Point: {len(sample_data)+1}")
    
    # Ensure drift detection chart is rendered correctly
    assert mock_altair_chart.call_count == 2  # There should be two charts in total (MSE + Drift)

