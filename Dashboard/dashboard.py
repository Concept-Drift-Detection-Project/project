import streamlit as st
import pandas as pd
import altair as alt

st.write("""
# Concept Drift Detector
""")

# Divide the page into two columns
left_col, right_col = st.columns(2)

# Left side
with left_col:
    st.write("### Upload Dataset and Select Model")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df)

        # Dropdown for selecting regression model
        model_options = ["Linear Regression", "Decision Tree Regression", 
                         "RandomForest Regression", "Support Vector Regression"]
        selected_model = st.selectbox("Select Regression Model:", model_options)

        st.button("Submit")

# Right side
with right_col:
    st.write("### Drift Detection Results")

    # Sample table with drift detector information
    drift_detector_options = ["ADWIN", "Page Hinkley", "DDM", "EDDM"]
    drift_detector = st.selectbox("Drift Detector:", drift_detector_options)
    
    # Example table data
    table_data = {
        "Drift Detector": [drift_detector],
        "False Alarms": [5],  
        "False Alarms Rate": [],
        "Drift Detection Delay": [10]  
    }
    table_df = pd.DataFrame(table_data)
    st.table(table_df)
    
    # Space for a plot
    st.write("### Example Plot")
    if uploaded_file is not None:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 0:
            plot_column = st.selectbox("Select Column to Plot:", numeric_columns)
            chart = alt.Chart(df).mark_bar().encode(
                alt.X(plot_column, bin=alt.Bin(maxbins=20)),
                y='count()'
            ).properties(
                title=f'Distribution of {plot_column}'
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No numeric columns available for plotting.")
