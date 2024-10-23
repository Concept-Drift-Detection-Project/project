import streamlit as st
import numpy as np
import pandas as pd
from river.datasets import synth
import altair as alt

def test_synthetic_dataset_generation():
    st.title("Synthetic Dataset Generation Test")

    # Parameters for synthetic dataset generation
    first_point = st.number_input("Enter the first drift point:", min_value=1000, value=7000, step=1000)
    second_point = st.number_input("Enter the second drift point:", min_value=first_point + 1000, value=11000, step=1000)

    # Button to generate dataset
    if st.button("Generate Synthetic Dataset"):
        # Generate the dataset
        dataset = synth.FriedmanDrift(
            drift_type='gra',
            position=(first_point, second_point),
            seed=42
        )

        # Initialize the data container
        data = []
        for i, (x, y) in enumerate(dataset):
            x_values = list(x.values())
            data.append(x_values + [y])
            if i >= second_point:  # Limiting to second_point samples for simplicity
                break

        # Define the column names and create the DataFrame
        column_names = [f'x{i}' for i in range(1, len(x_values) + 1)] + ['y']
        df = pd.DataFrame(data, columns=column_names)

        # Visualize the dataset
        st.subheader("Generated Synthetic Dataset")
        st.write(df)

        # Plotting the dataset
        chart_data = pd.DataFrame({
            'Index': np.arange(len(data)),
            'Target Variable (y)': [row[-1] for row in data]
        })

        chart = alt.Chart(chart_data).mark_line().encode(
            x='Index',
            y='Target Variable (y)'
        ).properties(
            width=600,
            height=300,
            title="Synthetic Dataset Target Variable Over Time"
        )
        st.altair_chart(chart, use_container_width=True)

        # Display drift points
        st.write(f"Drift occurs at points: {first_point} and {second_point}.")

