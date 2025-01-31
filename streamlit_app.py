import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model
import numpy as np

# Load your pre-trained models
lstm_model = load_model('lstm_model.h5')
xgboost_model = XGBRegressor()
xgboost_model.load_model('xgboost_model.json')

# Load the sales data
@st.cache
def load_data():
    return pd.read_csv('sales_data5.csv')

data = load_data()

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Sidebar for user inputs
st.sidebar.header('Filter Options')
product_names = data['product_name'].unique()
selected_product = st.sidebar.selectbox('Select Product', product_names)

locations = data['warehouse_location'].unique()
selected_location = st.sidebar.selectbox('Select Location', locations)

date_range = st.sidebar.date_input('Select Date Range', [data['date'].min(), data['date'].max()])

# Filter data based on user selection
filtered_data = data[
    (data['product_name'] == selected_product) &
    (data['warehouse_location'] == selected_location) &
    (data['date'] >= pd.to_datetime(date_range[0])) &
    (data['date'] <= pd.to_datetime(date_range[1]))
]

# Display filtered data
st.write('Filtered Sales Data')
st.write(filtered_data)

# Plot sales data
st.write('Sales Chart')
fig, ax = plt.subplots()
sns.lineplot(x='date', y='product_sales_quantity', data=filtered_data, ax=ax)
st.pyplot(fig)

# Sales prediction
st.sidebar.header('Sales Prediction')
prediction_date = st.sidebar.date_input('Select Date for Prediction')

if st.sidebar.button('Predict Sales'):
    # Prepare data for prediction
    # Assuming you have a function to preprocess the data for the models
    def preprocess_data_for_prediction(data, prediction_date):
        # Add your preprocessing steps here
        # For example, feature engineering, scaling, etc.
        return processed_data

    processed_data = preprocess_data_for_prediction(filtered_data, prediction_date)

    # Predict using LSTM
    lstm_prediction = lstm_model.predict(processed_data)

    # Predict using XGBoost
    xgboost_prediction = xgboost_model.predict(processed_data)

    # Display predictions
    st.write(f'LSTM Prediction for {prediction_date}: {lstm_prediction[0]}')
    st.write(f'XGBoost Prediction for {prediction_date}: {xgboost_prediction[0]}')
