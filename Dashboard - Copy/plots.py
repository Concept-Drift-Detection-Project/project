import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.write("""
# Concept Drift Detector
""")

@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info("Upload a file through config", icon="ℹ️")
    st.stop()

df = load_data(uploaded_file)
st.dataframe(df)

# Add some basic plots
st.write("## Data Distribution")
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Histogram
st.write("### Histogram")
for col in numeric_columns:
    fig, ax = plt.subplots()
    df[col].hist(bins=20, ax=ax)
    ax.set_title(f'Histogram of {col}')
    st.pyplot(fig)

# Boxplot
st.write("### Boxplot")
for col in numeric_columns:
    fig, ax = plt.subplots()
    df.boxplot(column=col, ax=ax)
    ax.set_title(f'Boxplot of {col}')
    st.pyplot(fig)

# Scatter plot (between first two numeric columns)
if len(numeric_columns) > 1:
    st.write("### Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(df[numeric_columns[0]], df[numeric_columns[1]])
    ax.set_xlabel(numeric_columns[0])
    ax.set_ylabel(numeric_columns[1])
    ax.set_title(f'Scatter Plot of {numeric_columns[0]} vs {numeric_columns[1]}')
    st.pyplot(fig)
