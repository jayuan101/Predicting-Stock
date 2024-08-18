import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title and Description
st.title("Interactive Stock Data Viewer")
st.write("Upload a CSV file containing stock data to interact with it.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the data
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Select Columns to Display
    columns = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist())
    st.write("### Filtered Data")
    st.dataframe(df[columns])

    # Filter by Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        start_date = st.date_input("Start date", df['Date'].min())
        end_date = st.date_input("End date", df['Date'].max())
        
        filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
        st.write(f"### Data from {start_date} to {end_date}")
        st.dataframe(filtered_df)

        # Plot the data
        st.write("### Stock Price Over Time")
        selected_column = st.selectbox("Select a column to plot", df.columns[1:], index=1)  # Assuming Date is the first column
        plt.figure(figsize=(10, 5))
        plt.plot(filtered_df['Date'], filtered_df[selected_column])
        plt.xlabel("Date")
        plt.ylabel(selected_column)
        plt.title(f"{selected_column} over Time")
        st.pyplot(plt)
    else:
        st.write("The uploaded CSV file does not contain a 'Date' column. Please upload a valid stock data file.")
else:
    st.write("Please upload a CSV file to proceed.")
