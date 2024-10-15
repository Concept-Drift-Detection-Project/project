import sys
import os 
from unittest.mock import patch, MagicMock
import pandas as pd 

# Add the project root directory to Python's path, so the test can find home.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import choice 

@patch('default.st.altair_chart')
@patch('default.st.number_input')
@patch('default.st.selectbox')
@patch('default.st.button')
@patch('default.st.columns')
@patch('default.st.write')
@patch('default.st.title')
@patch('default.st.subheader')
def test_choose(mock_subheader, mock_title, mock_write, mock_columns, mock_button, 
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
    mock_number_input.side_effect = [8000, 11000]  # Simulate drift points input (first_point = 48000, second_point = 75000)
    
    # Simulate the button press
    mock_button.return_value = True  # Simulate "Run Drift Detection" button press
    
    # Run the function
    choice.choose() 

    # Assert that Streamlit components were called correctly
    mock_title.assert_called_once_with('Concept Drift Detection in a Synthetic Dataset')
    
    mock_subheader.assert_any_call("Drift Detection Results Table")
    
    mock_number_input.assert_any_call("Enter the first drift point:", min_value=1000, value=8000, step=1000)
    mock_number_input.assert_any_call("Enter the second drift point:", min_value=9000, value=11000, step=1000)  # Corrected based on logic
    
    # Ensure that a button was clicked
    mock_button.assert_called_once_with("Run Drift Detection")

    mock_subheader.assert_any_call("Results taken by optimum configuration")
    mock_subheader.assert_any_call("Results taken by default configuration")

     # Collect all DataFrames passed to st.write()
    write_calls = [call[0][0] for call in mock_write.call_args_list if isinstance(call[0][0], pd.DataFrame)]
    
    # Log the actual DataFrame passed to st.write()
    print("Actual DataFrames passed to st.write():")
    for df in write_calls:
        print(df)  # Print the actual DataFrame to inspect its structure and content

    # Ensure 8 DataFrames were written
    assert len(write_calls) == 8, f"Expected 8 DataFrames to be written, but found {len(write_calls)}."

    # Define the expected structure of the DataFrame, update based on actual function logic
    expected_drift_results = {
        'False Alarms': [0],  # Replace with expected values
        'False Alarm Rate': [0.0],  # Replace with expected values
        'Detection Delay': [None],  # Replace with expected values
        'Average Measure': [0]  # Replace with expected values
    }
    
    expected_df = pd.DataFrame(expected_drift_results)
    
    # Ensure that the structure of all DataFrames matches expected_df
    for call in write_calls:
        print("Comparing DataFrame structure:")
        print(f"Expected columns: {expected_df.columns}")
        print(f"Actual columns: {call.columns}")
        
        assert expected_df.columns.equals(call.columns) and expected_df.dtypes.equals(call.dtypes), \
            f"Expected DataFrame structure does not match for call: {call}" 