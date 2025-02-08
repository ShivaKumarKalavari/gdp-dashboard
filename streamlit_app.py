import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import requests
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Download the LSTM model from GitHub
url1 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/model_X.keras'
response = requests.get(url1)
with open('model_X.keras', 'wb') as f:
    f.write(response.content)

# Load the trained LSTM model
model_path = 'model_X.keras'  # Update with the correct model path
lstm_model = tf.keras.models.load_model(model_path)


# Download the sales data from GitHub
url2 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/data/sales_data_new.csv'
response = requests.get(url2)
with open('sales_data_new.csv', 'wb') as f:
    f.write(response.content)

# Load sales data
sales_data_path = 'sales_data_new.csv'  # Update with the correct data path
data = pd.read_csv(sales_data_path)

# Sidebar for user inputs
st.sidebar.header('Filter Options')
product_names = data['product_name'].unique()
selected_product = st.sidebar.selectbox('Select Product', product_names)

locations = data['warehouse_location'].unique()
selected_location = st.sidebar.selectbox('Select Location', locations)

# Sidebar selection for year and month range
st.sidebar.header('Filter Options')
selected_year_range = st.sidebar.slider('Select Year Range', int(data['year'].min()), int(data['year'].max()), (int(data['year'].min()), int(data['year'].max())))
selected_month_range = st.sidebar.slider('Select Month Range', 1, 12, (1, 12))

# Filter data based on user selection
filtered_data = data[
    (data['product_name'] == selected_product) &
    (data['warehouse_location'] == selected_location) &
    (data['year'] >= selected_year_range[0]) & (data['year'] <= selected_year_range[1]) &
    (data['month'] >= selected_month_range[0]) & (data['month'] <= selected_month_range[1])
]

# Display filtered data
st.write('Filtered Sales Data')
st.write(filtered_data)

# Plot sales data
st.write('Sales Chart')
fig, ax = plt.subplots()
sns.lineplot(x='month', y='product_sales_quantity', hue='year', data=filtered_data, ax=ax)
st.pyplot(fig)


# Encode categorical features
label_enc_category = LabelEncoder()
label_enc_location = LabelEncoder()
data['product_category_encoded'] = label_enc_category.fit_transform(data['product_category'])
data['warehouse_location_encoded'] = label_enc_location.fit_transform(data['warehouse_location'])

# Normalize sales quantity
scaler = MinMaxScaler()
data['product_sales_quantity'] = scaler.fit_transform(data[['product_sales_quantity']])

# Sidebar inputs
st.sidebar.header('Predict Sales')
selected_year = st.sidebar.selectbox('Select Year', sorted(data['year'].unique()))
selected_month = st.sidebar.selectbox('Select Month', sorted(data['month'].unique()))
selected_category = st.sidebar.selectbox('Select Product Category', data['product_category'].unique())
selected_location = st.sidebar.selectbox('Select Warehouse Location', data['warehouse_location'].unique())

# Encode user inputs
category_encoded = label_enc_category.transform([selected_category])[0]
location_encoded = label_enc_location.transform([selected_location])[0]

# Prepare input for LSTM
input_features = np.array([[category_encoded, location_encoded, selected_month, selected_year]])
input_features = np.reshape(input_features, (1, 1, input_features.shape[1]))

# Predict sales
if st.sidebar.button('Predict Sales'):
    prediction = lstm_model.predict(input_features)
    predicted_sales = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    st.write(f'Predicted Sales Quantity for {selected_month}/{selected_year}: {predicted_sales:.2f}')
