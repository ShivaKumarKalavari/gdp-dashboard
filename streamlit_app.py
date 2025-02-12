import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import requests
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Login System
users = {"admin": "password123", "user": "pass456"}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("Login to Sales Forecast Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")

if not st.session_state.logged_in:
    login()
    st.stop()


# Download the sales data from GitHub
url1 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/data/sales_data_new.csv'
response = requests.get(url1)
with open('sales_data_new.csv', 'wb') as f:
    f.write(response.content)

# Load sales data
path = 'sales_data_new.csv'
data = pd.read_csv(path)

# Convert date
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
#################
# Create complete grid of dates, categories, and warehouses
all_dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='MS')
categories = data['product_category'].unique()
warehouses = data['warehouse_location'].unique()
index = pd.MultiIndex.from_product([all_dates, categories, warehouses], names=['date', 'product_category', 'warehouse_location'])
df_grouped = data.groupby(['date', 'product_category', 'warehouse_location'])['product_sales_quantity'].sum().reindex(index, fill_value=0).reset_index()

# Feature Engineering
df_grouped.sort_values(['product_category', 'warehouse_location', 'date'], inplace=True)
for lag in [1, 2, 3]:
    df_grouped[f'lag_{lag}'] = df_grouped.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].shift(lag)

window_sizes = [3, 6]
for window in window_sizes:
    df_grouped[f'rolling_mean_{window}'] = df_grouped.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df_grouped[f'rolling_std_{window}'] = df_grouped.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].transform(lambda x: x.rolling(window, min_periods=1).std())

df_grouped['month_sin'] = np.sin(2 * np.pi * df_grouped['date'].dt.month / 12)
df_grouped['month_cos'] = np.cos(2 * np.pi * df_grouped['date'].dt.month / 12)
df_grouped['year'] = df_grouped['date'].dt.year
df_grouped['time_idx'] = (df_grouped['date'].dt.year - df_grouped['date'].dt.year.min()) * 12 + (df_grouped['date'].dt.month - df_grouped['date'].dt.month.min())
df_grouped.fillna(0, inplace=True)

# One-hot encode categorical variables
df_grouped = pd.get_dummies(df_grouped, columns=['product_category', 'warehouse_location'], drop_first=False)


# Define features and target
X = df_grouped.drop(columns=['date', 'product_sales_quantity'])
y = df_grouped['product_sales_quantity']
###################

# Creating the analytics dashboard
st.title("Sales Analytics Dashboard")
data = data.drop(columns=['date'])

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

st.subheader('Filtered Sales Data')
st.write(data_filtered)

# Sales Chart
st.subheader('Sales Chart')
fig, ax = plt.subplots()
sns.lineplot(x='month', y='product_sales_quantity', hue='year', data=data_filtered, ax=ax)
st.pyplot(fig)

# Pie Chart for Market Share

# Sidebar filters for Market Share Analysis
st.sidebar.header("Market Share Filter Options")
selected_location_pie = st.sidebar.selectbox("Select Warehouse Location for Market Share", data['warehouse_location'].unique(), key="market_share_location")
selected_year_range_pie = st.sidebar.slider("Select Year Range for Market Share", int(data['year'].min()), int(data['year'].max()), 
                                            (int(data['year'].min()), int(data['year'].max())), key="market_share_year")
selected_month_range_pie = st.sidebar.slider("Select Month Range for Market Share", 1, 12, (1, 12), key="market_share_month")

# Filter the dataset based on selected location and time range
data_filtered_pie = data[
    (data['warehouse_location'] == selected_location_pie) &
    (data['year'] >= selected_year_range_pie[0]) & (data['year'] <= selected_year_range_pie[1]) &
    (data['month'] >= selected_month_range_pie[0]) & (data['month'] <= selected_month_range_pie[1])
]

# Update subheader dynamically
st.subheader(f"Market Share Analysis for {selected_location_pie} ({selected_year_range_pie[0]} - {selected_year_range_pie[1]}, Months {selected_month_range_pie[0]} - {selected_month_range_pie[1]})")

# Generate Market Share Pie Chart dynamically
if not data_filtered_pie.empty:
    category_share = data_filtered_pie.groupby("product_category")['product_sales_quantity'].sum()

    # Ensure there is valid data to plot
    if category_share.sum() > 0:
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(category_share, labels=category_share.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
        ax_pie.axis("equal")  # Ensures pie chart is circular
        st.pyplot(fig_pie)
    else:
        st.warning("No sales data available for the selected filters. Try a different time range or location.")
else:
    st.warning("No data available for the selected warehouse and time range.")




# Download the model from GitHub
url2 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/xgboost_model.json'
response = requests.get(url2)
with open('sales_data_new.csv', 'wb') as f:
    f.write(response.content)

# Load trained XGBoost model
model_path = 'xgboost_model.json'
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model_path)

# Sidebar for prediction
st.sidebar.header('Predict Sales')
selected_year = st.sidebar.selectbox('Select Year', [i for i in range(data['year'].max()+1,data['year'].max()+5)])
selected_month = st.sidebar.selectbox('Select Month', [i for i in range(1,13)])
selected_category = st.sidebar.selectbox('Select Product Category', data['product_category'].unique())
selected_location = st.sidebar.selectbox('Select Warehouse Location', data['warehouse_location'].unique())

# Predict sales
if st.sidebar.button('Predict Sales'):
    # Normalize user inputs to match column format
    formatted_category = f'product_category_{selected_category}'
    formatted_warehouse = f'warehouse_location_{selected_location}'
    
    # Ensure the mask is a valid boolean series
    mask = (df_grouped[formatted_category] == 1) & (df_grouped[formatted_warehouse] == 1)
    
    if not mask.any():
        st.error("No data available for the selected category and warehouse.")
    
    hist_data = df_grouped[mask].sort_values('date')
    last_date = hist_data['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end=f'{selected_year}-{selected_month}-01', freq='MS')
    
    # Extract initial lags and sales
    current_sales = hist_data['product_sales_quantity'].iloc[-3:].tolist()
    predictions = []
    
    for date in future_dates:
        features = {
            'year': date.year,
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'time_idx': (date.year - df_grouped['date'].dt.year.min()) * 12 + (date.month - 1),
            formatted_category: 1,
            formatted_warehouse: 1,
            'lag_1': current_sales[-1] if len(current_sales) >= 1 else 0,
            'lag_2': current_sales[-2] if len(current_sales) >= 2 else 0,
            'lag_3': current_sales[-3] if len(current_sales) >= 3 else 0,
            'rolling_mean_3': np.mean(current_sales[-3:]) if len(current_sales) >= 3 else np.mean(current_sales),
            'rolling_std_3': np.std(current_sales[-3:]) if len(current_sales) >= 3 else np.std(current_sales) if len(current_sales) > 0 else 0,
            'rolling_mean_6': np.mean(current_sales[-6:]) if len(current_sales) >= 6 else np.mean(current_sales),
            'rolling_std_6': np.std(current_sales[-6:]) if len(current_sales) >= 6 else np.std(current_sales) if len(current_sales) > 0 else 0
        }
        
        # Create feature DataFrame ensuring correct column order
        features_df = pd.DataFrame([features], columns=X.columns).fillna(0)
        pred = xgb_model.predict(features_df)[0]
        predictions.append((date, pred))
        current_sales.append(pred)
        
    st.subheader("Sales Forecast Analytics")
    
    # Generate forecast DataFrame and plot
    future_df = pd.DataFrame(predictions, columns=['date', 'predicted_sales'])

    st.write(f"The forecated sales quantity for the category '{selected_category}' in the location '{selected_location}' for the year '{selected_year}' and month '{selected_month}' is predicted around:",future_df.tail(1)['predicted_sales'].values[0])
    st.write("\n\n")
    st.write("If you want the 'Forecared sales' upto the selected date :\n")
    st.write(future_df)
    
