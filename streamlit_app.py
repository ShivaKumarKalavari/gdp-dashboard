import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import requests

# Download the sales data from GitHub
url1 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/data/sales_data_new.csv'
response = requests.get(url1)
with open('sales_data_new.csv', 'wb') as f:
    f.write(response.content)
    
# Load Data
@st.cache_data
def load_data():
    file_path = "sales_data_new.csv"
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df[df['date'] <= '2024-10-01']
    return df

df = load_data()

# Process Data
def prepare_data(df):
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')
    categories = df['product_category'].unique()
    warehouses = df['warehouse_location'].unique()
    index = pd.MultiIndex.from_product([all_dates, categories, warehouses], names=['date', 'product_category', 'warehouse_location'])
    df_grouped = df.groupby(['date', 'product_category', 'warehouse_location'])['product_sales_quantity'].sum().reindex(index, fill_value=0).reset_index()

    df_grouped.sort_values(['product_category', 'warehouse_location', 'date'], inplace=True)
    for lag in [1, 2, 3]:
        df_grouped[f'lag_{lag}'] = df_grouped.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].shift(lag)

    for window in [3, 6]:
        df_grouped[f'rolling_mean_{window}'] = df_grouped.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df_grouped[f'rolling_std_{window}'] = df_grouped.groupby(['product_category', 'warehouse_location'])['product_sales_quantity'].transform(lambda x: x.rolling(window, min_periods=1).std())

    df_grouped['month_sin'] = np.sin(2 * np.pi * df_grouped['date'].dt.month / 12)
    df_grouped['month_cos'] = np.cos(2 * np.pi * df_grouped['date'].dt.month / 12)
    df_grouped['year'] = df_grouped['date'].dt.year
    df_grouped['time_idx'] = (df_grouped['date'].dt.year - df_grouped['date'].dt.year.min()) * 12 + (df_grouped['date'].dt.month - df_grouped['date'].dt.month.min())
    df_grouped.fillna(0, inplace=True)

    return pd.get_dummies(df_grouped, columns=['product_category', 'warehouse_location'], drop_first=False)

df_grouped = prepare_data(df)

# Download the sales data from GitHub
url2 = 'https://github.com/ShivaKumarKalavari/gdp-dashboard/raw/main/xgboost_model.json'
response = requests.get(url2)
with open('xgboost_model.json', 'wb') as f:
    f.write(response.content)

# Load Model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")  # Ensure this path is correct
    return model

model = load_model()

# Streamlit UI
st.title("Sales Forecasting with XGBoost")
st.write("Select inputs to predict future sales.")

# User Inputs
input_year = st.number_input("Enter the year:", min_value=2015, max_value=2025, value=2024)
input_month = st.number_input("Enter the month (1-12):", min_value=1, max_value=12, value=10)
input_category = st.selectbox("Select product category:", df['product_category'].unique())
input_warehouse = st.selectbox("Select warehouse:", df['warehouse_location'].unique())

formatted_category = f'product_category_{input_category}'
formatted_warehouse = f'warehouse_location_{input_warehouse}'

# Check if the columns exist
if formatted_category not in df_grouped.columns or formatted_warehouse not in df_grouped.columns:
    st.error("Invalid category or warehouse. Please check the input selection.")
else:
    # Filter the historical data
    mask = (df_grouped[formatted_category] == 1) & (df_grouped[formatted_warehouse] == 1)
    hist_data = df_grouped[mask].sort_values('date')

    if hist_data.empty:
        st.warning("No data available for the selected category and warehouse.")
    else:
        last_date = hist_data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), end=f'{input_year}-{input_month}-01', freq='MS')

        # Extract initial lags
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

            features_df = pd.DataFrame([features], columns=df_grouped.drop(columns=['date', 'product_sales_quantity']).columns).fillna(0)
            pred = model.predict(features_df)[0]
            predictions.append((date, pred))
            current_sales.append(pred)

        future_df = pd.DataFrame(predictions, columns=['date', 'predicted_sales'])

        # Display results
        st.subheader("Forecasted Sales")
        st.write(future_df)

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hist_data['date'], hist_data['product_sales_quantity'], label='Historical Sales', marker='o')
        ax.plot(future_df['date'], future_df['predicted_sales'], label='Forecast', linestyle='--', marker='x', color='red')
        ax.axvline(last_date, color='gray', linestyle='--', label='Forecast Start')
        ax.set_title(f"Sales Forecast for {input_category} in {input_warehouse}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales Quantity")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
