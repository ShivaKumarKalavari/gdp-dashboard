import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import requests
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load sales data
url = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/data/sales_data_new.csv'
data = pd.read_csv(url)

# Convert date
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

# Feature Engineering
for lag in [1, 2, 3]:
    data[f'lag_{lag}'] = data.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].shift(lag)

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data.fillna(0, inplace=True)

# One-hot encoding
label_enc_category = LabelEncoder()
label_enc_location = LabelEncoder()
data['product_category_encoded'] = label_enc_category.fit_transform(data['product_category'])
data['warehouse_location_encoded'] = label_enc_location.fit_transform(data['warehouse_location'])

# Normalize sales quantity
scaler = MinMaxScaler()
data['product_sales_quantity'] = scaler.fit_transform(data[['product_sales_quantity']])

# Sidebar filters
st.sidebar.header('Filter Options')
selected_product = st.sidebar.selectbox('Select Product', data['product_name'].unique())
selected_location = st.sidebar.selectbox('Select Location', data['warehouse_location'].unique())
selected_year_range = st.sidebar.slider('Select Year Range', int(data['year'].min()), int(data['year'].max()), (int(data['year'].min()), int(data['year'].max())))
selected_month_range = st.sidebar.slider('Select Month Range', 1, 12, (1, 12))

# Filtered Data
data_filtered = data[(data['product_name'] == selected_product) &
                     (data['warehouse_location'] == selected_location) &
                     (data['year'] >= selected_year_range[0]) & (data['year'] <= selected_year_range[1]) &
                     (data['month'] >= selected_month_range[0]) & (data['month'] <= selected_month_range[1])]

st.write('Filtered Sales Data')
st.write(data_filtered)

# Sales Chart
st.write('Sales Chart')
fig, ax = plt.subplots()
sns.lineplot(x='month', y='product_sales_quantity', hue='year', data=data_filtered, ax=ax)
st.pyplot(fig)

# Load trained XGBoost model
model_url = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/xgboost_model.json'
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model_url)

# Sidebar for prediction
st.sidebar.header('Predict Sales')
selected_year = st.sidebar.selectbox('Select Year', sorted(data['year'].unique()))
selected_month = st.sidebar.selectbox('Select Month', sorted(data['month'].unique()))
selected_category = st.sidebar.selectbox('Select Product Category', data['product_category'].unique())

# Encode user inputs
category_encoded = label_enc_category.transform([selected_category])[0]
location_encoded = label_enc_location.transform([selected_location])[0]

# Prepare input for prediction
input_features = np.array([[category_encoded, location_encoded, selected_month, selected_year, np.sin(2 * np.pi * selected_month / 12), np.cos(2 * np.pi * selected_month / 12)]]).astype(float)

# Predict sales
if st.sidebar.button('Predict Sales'):
    prediction = xgb_model.predict(input_features)
    predicted_sales = scaler.inverse_transform([[prediction[0]]])[0][0]
    st.write(f'Predicted Sales Quantity for {selected_month}/{selected_year}: {predicted_sales:.2f}')
