import sys
import os 
from unittest.mock import patch, MagicMock
import pandas as pd 

# Add the project root directory to Python's path, so the test can find home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import default  


@patch('default.st.altair_chart')
@patch('default.st.number_input')
@patch('default.st.selectbox')
@patch('default.st.button')
@patch('default.st.columns')
@patch('default.st.write')
@patch('default.st.title')
@patch('default.st.subheader')
def test_compare(mock_subheader, mock_title, mock_write, mock_columns, mock_button, 
                 mock_selectbox, mock_number_input, mock_altair_chart):
    
    # Mocking the return values for the UI elements
    mock_col1 = MagicMock()
    mock_col2 = MagicMock()
    mock_columns.return_value = (mock_col1, mock_col2)  # Mock columns to return two mock objects
    
    mock_title.return_value = None
    mock_write.return_value = None
    mock_subheader.return_value = None
    
    # Mocking the values for dropdowns and inputs
    mock_selectbox.side_effect = ["Linear Regressor"]  # Simulate model choice
    mock_number_input.side_effect = [7000, 11000]  # Simulate drift points input
    
    # Simulate the button press
    mock_button.return_value = True  # Simulate "Run Drift Detection" button press
    
    # Run the function
    default.compare()

    # Assert that Streamlit components were called correctly
    mock_title.assert_called_once_with('Concept Drift Detection in a Synthetic Dataset')
    
    mock_subheader.assert_any_call("Drift Detection Results Table")
    mock_selectbox.assert_any_call("Select the regression model:", 
                                    ("Linear Regressor", "SVM Regressor", "Decision Tree Regressor", "Random Forest Regressor"))
    mock_number_input.assert_any_call("Enter the first drift point:", min_value=1000, value=7000, step=1000)
    mock_number_input.assert_any_call("Enter the second drift point:", min_value=8000, value=11000, step=1000)

    # Ensure that a button was clicked
    mock_button.assert_called_once_with("Run Drift Detection")

    mock_subheader.assert_any_call("Drift Detection Indicator")

    # Verify Altair charts were rendered
    assert mock_altair_chart.call_count == 6  # Check that two charts are created

    