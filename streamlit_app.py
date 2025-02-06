import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import tensorflow as tf
import numpy as np
import requests

# Download the LSTM model from GitHub
url1 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/lstm_model.h5'
response = requests.get(url1)
with open('lstm_model.h5', 'wb') as f:
    f.write(response.content)

# Load the LSTM model
try:
    lstm_model = tf.keras.models.load_model('lstm_model.h5')
    lstm_model.compile(optimizer='adam', loss='mse') 
except Exception as e:
    st.error(f"Error loading LSTM model: {e}")
    st.stop()

# Download the XGBoost model from GitHub
url2 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/xgboost_model.json'
response = requests.get(url2)
with open('xgboost_model.json', 'wb') as f:
    f.write(response.content)

# Load the XGBoost model
xgboost_model = XGBRegressor()
xgboost_model.load_model('xgboost_model.json')

# Download the sales data from GitHub
url3 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/data/sales_data.csv'
response = requests.get(url3)
with open('sales_data.csv', 'wb') as f:
    f.write(response.content)

# Load the sales data
data = pd.read_csv('sales_data.csv')

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
    def preprocess_data_for_prediction(data, prediction_date):
        # Example: Extract year, month, and day from the prediction date
        processed_data = np.array([[prediction_date.year, prediction_date.month, prediction_date.day]])
        return processed_data

    processed_data = preprocess_data_for_prediction(filtered_data, prediction_date)

    # Predict using LSTM
    lstm_prediction = lstm_model.predict(processed_data)

    # Predict using XGBoost
    xgboost_prediction = xgboost_model.predict(processed_data)

    # Display predictions
    st.write(f'LSTM Prediction for {prediction_date}: {lstm_prediction[0]}')
    st.write(f'XGBoost Prediction for {prediction_date}: {xgboost_prediction[0]}')
