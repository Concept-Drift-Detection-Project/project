import streamlit as st
import pandas as pd


def show():

    st.title("Best Configurations for Drift Detection Methods")

    st.write("""
    This page displays the best configuration values for the drift detection methods.
    """)


    # Best configurations values for Linear Regressor

    st.subheader("Linear Regressor")

    data_LR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 1.65", "Alpha = 0.90", "Clock = 91", "Lambda = 32.0"],
        "Parameter 2": ["Drift Level = 1.7", "Beta = 0.85", "Min Window Size = 56", "Alpha = 0.9099"],
        "Parameter 3": ["Min Instances = 330", "Level = 1.95", "Min Num Instances = 10", "Min Num Instances = 88"],
        "Parameter 4": ["", "Min Instances = 50", "Memory = 7", "delta = 0.005"],
        "Parameter 5": ["", "", "delta = 0.002", ""]
    }

    df_LR = pd.DataFrame(data_LR)

    # Display table
    st.table(df_LR)


    # Best configurations values for Decision Tree Regressor

    st.subheader("Decision Tree Regressor")

    data_DTR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 2.65", "Alpha = 0.90", "Clock = 19", "Lambda = 61.0"],
        "Parameter 2": ["Drift Level = 2.7", "Beta = 0.85", "Min Window Size = 90", "Alpha = 0.7649"],
        "Parameter 3": ["Min Instances = 250", "Level = 1.55", "Min Num Instances = 96", "Min Num Instances = 75"],
        "Parameter 4": ["", "Min Instances = 170", "Memory = 15", "delta = 0.005"],
        "Parameter 5": ["", "", "delta = 0.002", ""]
    }

    df_DTR = pd.DataFrame(data_DTR)

    # Display table
    st.table(df_DTR)


    # Best configurations values for Random Forest Regressor

    st.subheader("Random Forest Regressor")

    data_RFR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 5.05", "Alpha = 0.80", "Clock = 1", "Lambda = 12.0"],
        "Parameter 2": ["Drift Level = 5.1", "Beta = 0.75", "Min Window Size = 52", "Alpha = 0.9899"],
        "Parameter 3": ["Min Instances = 30", "Level = 1.85", "Min Instances = 10", "Min Num Instances = 74"],
        "Parameter 4": ["", "Min Instances = 110", "Memory = 5", "delta = 0.005"],
        "Parameter 5": ["", "", "delta = 0.002", ""]
    }

    df_RFR = pd.DataFrame(data_RFR)

    # Display table
    st.table(df_RFR)


    # Best configurations values for Support Vector Regressor

    st.subheader("Support Vector Regressor")

    data_SVR = {
        "Method": ["DDM", "EDDM", "ADWIN", "Page Hinkley"],
        "Parameter 1": ["Warning Level = 2.45", "Alpha = 1.0", "Clock = 91", "Lambda = 19.0"],
        "Parameter 2": ["Drift Level = 2.5", "Beta = 0.95", "Min Window Size = 72", "Alpha = 0.9249"],
        "Parameter 3": ["Min Instances = 90", "Level = 1.0", "Min Num Instances = 10", "Min Num Instances = 81"],
        "Parameter 4": ["", "Min Instances = 110", "Memory = 7", "delta = 0.005"],
        "Parameter 5": ["", "", "delta = 0.002", ""]
    }

    df_SVR = pd.DataFrame(data_SVR)

    # Display table
    st.table(df_SVR)