import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

def load_data(file):
    try:
        data = pd.read_csv(file)
        file_type = 'csv'
    except:
        try:
            data = pd.read_excel(file)
            file_type = 'excel'
        except:
            raise ValueError("Invalid file format. Please upload a CSV or Excel file.")
    return data, file_type

def handle_missing_values(data, strategy):
    if strategy == 'drop':
        data = data.dropna()
    elif strategy == 'mean':
        data = data.fillna(data.mean())
    elif strategy == 'median':
        data = data.fillna(data.median())
    elif strategy == 'mode':
        data = data.fillna(data.mode().iloc[0])
    return data

def encode_categorical_data(data, columns, encoding_type):
    if encoding_type == 'label':
        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
    elif encoding_type == 'one-hot':
        encoder = OneHotEncoder(sparse=False)
        encoded_data = pd.DataFrame(encoder.fit_transform(data[columns]))
        encoded_data.columns = encoder.get_feature_names(columns)
        data = pd.concat([data, encoded_data], axis=1)
        data = data.drop(columns, axis=1)
    return data

def scale_numerical_data(data, columns, scaling_type):
    if scaling_type == 'standard':
        scaler = StandardScaler()
    elif scaling_type == 'minmax':
        scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def main():
    st.title("Data Preprocessing App")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        data, file_type = load_data(uploaded_file)

        # Show preview of the data
        st.subheader("Data Preview")
        st.write(data.head())

        # Handling missing values
        missing_values_option = st.selectbox("Handle missing values", ["No", "Drop rows", "Mean", "Median", "Mode"])
        if missing_values_option != "No":
            strategy = missing_values_option.lower().split()[1]
            data = handle_missing_values(data, strategy)

        # Encoding categorical data
        encode_categorical = st.checkbox("Encode categorical data")
        if encode_categorical:
            categorical_columns = st.text_input("Enter column names for categorical data (comma-separated)").replace(" ", "").split(",")
            encoding_type = st.radio("Select encoding type", ["Label", "One-hot"])
            data = encode_categorical_data(data, categorical_columns, encoding_type.lower())

        # Scaling numerical data
        scale_numerical = st.checkbox("Scale numerical data")
        if scale_numerical:
            numerical_columns = st.text_input("Enter column names for numerical data (comma-separated)").replace(" ", "").split(",")
            scaling_type = st.radio("Select scaling type", ["Standard", "MinMax"])
            data = scale_numerical_data(data, numerical_columns, scaling_type.lower())

        # Download preprocessed data
        st.subheader("Download Preprocessed Data")
        preprocessed_data = data.to_csv(index=False)
        st.download_button(
            label="Download Preprocessed Data",
            data=preprocessed_data,
            file_name=f"preprocessed_data.{file_type}",
            mime=f"text/{file_type}",
        )

if __name__ == "__main__":
    main()