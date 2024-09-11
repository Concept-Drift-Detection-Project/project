import streamlit as st
import pandas as pd


def show():

    st.title("Home: Best Configurations for Drift Detection Methods")

    st.write("""
    This page displays the best configuration values for the drift detection methods.
    """)


    # Best configurations values for Linear Regressor

    st.subheader("Linear Regressor")

    data_LR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 1.65", "Alpha = 0.90", "Clock = 128", "Lambda = 10.0"],
        "Parameter 2": ["Drift Level = 1.7", "Beta = 0.85", "Min Window Size = 5", "Alpha = 0.9999"],
        "Parameter 3": ["Min Instances = 330", "Level = 1.95", "Min Instances = 10", "Min Instances = 30"],
        "Parameter 4": ["", "Level = 2.0", "Min Instances = 50", "Min Instances = 30"]

    }

    df_LR = pd.DataFrame(data_LR)

    # Display table
    st.table(df_LR)


    # Best configurations values for Decision Tree Regressor

    st.subheader("Decision Tree Regressor")

    data_DTR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 1.65", "Alpha = 0.90", "Clock = 128", "Lambda = 10.0"],
        "Parameter 2": ["Drift Level = 1.7", "Beta = 0.85", "Min Window Size = 5", "Alpha = 0.9999"],
        "Parameter 3": ["Min Instances = 330", "Level = 1.95", "Min Instances = 10", "Min Instances = 30"],
        "Parameter 4": ["", "Level = 2.0", "Min Instances = 50", "Min Instances = 30"]

    }

    df_DTR = pd.DataFrame(data_DTR)

    # Display table
    st.table(df_DTR)


    # Best configurations values for Random Forest Regressor

    st.subheader("Random Forest Regressor")

    data_RFR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 1.65", "Alpha = 0.90", "Clock = 128", "Lambda = 10.0"],
        "Parameter 2": ["Drift Level = 1.7", "Beta = 0.85", "Min Window Size = 5", "Alpha = 0.9999"],
        "Parameter 3": ["Min Instances = 330", "Level = 1.95", "Min Instances = 10", "Min Instances = 30"],
        "Parameter 4": ["", "Level = 2.0", "Min Instances = 50", "Min Instances = 30"]

    }

    df_RFR = pd.DataFrame(data_RFR)

    # Display table
    st.table(df_RFR)


    # Best configurations values for Support Vector Regressor

    st.subheader("Support Vector Regressor")

    data_SVR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 1.65", "Alpha = 0.90", "Clock = 128", "Lambda = 10.0"],
        "Parameter 2": ["Drift Level = 1.7", "Beta = 0.85", "Min Window Size = 5", "Alpha = 0.9999"],
        "Parameter 3": ["Min Instances = 330", "Level = 1.95", "Min Instances = 10", "Min Instances = 30"],
        "Parameter 4": ["", "Level = 2.0", "Min Instances = 50", "Min Instances = 30"]

    }

    df_SVR = pd.DataFrame(data_SVR)

    # Display table
    st.table(df_SVR)