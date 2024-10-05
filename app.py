from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from flask_cors import CORS
import os
app = Flask(__name__)
import pickle
cors = CORS(app)

import requests
from bs4 import BeautifulSoup
import pandas as pd
# Memuat model dari file pickle
with open('catboost_model.pkl', 'rb') as file:
    catboost_model = pickle.load(file)
data_beras = pd.read_csv("data/beras.csv")
selected_model = load_model(f'model/model_beras.h5')
# Function to perform future forecasting
def perform_forecasting(model, series, target_date):
    
    future_forecast = []

    # Calculate the number of days to predict
    days_to_predict = (target_date - pd.to_datetime("2024-10-04")).days
    print(days_to_predict)
    input_data = series[-60:days_to_predict-60][np.newaxis]
    for _ in range(days_to_predict):
        prediction = model.predict(input_data)
        future_forecast.append(prediction[0, 0])
        input_data = np.append(input_data[:, 1:, :], [[prediction[0, 0]]], axis=1)
    
    return future_forecast,days_to_predict

def perform_forecasting_seasonal(model, series, days_to_predict):
    
    future_forecast = []

    # Calculate the number of days to predict
    print(days_to_predict)
    input_data = series[-60:days_to_predict-60][np.newaxis]
    for _ in range(days_to_predict):
        prediction = model.predict(input_data)
        future_forecast.append(prediction[0, 0])
        input_data = np.append(input_data[:, 1:, :], [[prediction[0, 0]]], axis=1)
    
    return future_forecast,days_to_predict

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    target_date_str = request.json['tanggal']
    target_date = pd.to_datetime(target_date_str, format="%d-%m-%Y")
    
    # Perform future forecasting until the target date
    price_data = np.array(data_beras['Beras']).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = scaler.fit_transform(price_data.reshape(-1, 1))
    future_forecast,days = perform_forecasting(selected_model, series, target_date)

    # Convert the result to a list before sending it as JSON
    result = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten().tolist()
    
    future_dates = pd.date_range(start=data_beras['index'].iloc[-1], periods=days+1)[1:]
    # future_dates = future_dates.strftime('%Y-%m-%d')
    future_dates_str = future_dates[-30:].astype(str).tolist()
    data_beras['index'] = pd.to_datetime(data_beras['index'], format="%d-%m-%Y")
    data_beras['index'] = data_beras['index'].dt.strftime('%Y-%m-%d')
    df_beras = data_beras[['index','Beras']].tail(1000)

    return jsonify({'historical_data':{'date':df_beras['index'].tolist(), 'value':df_beras['Beras'].tolist()},
                    'forecasting_data':{'date':future_dates_str, 'value':result}})

@app.route('/predict_seasonal', methods=['POST'])
def predict_seasonal():
    # Get data from the request
    target_date_str = request.json['seasonal']
    
    # Perform future forecasting until the target date
    price_data = np.array(data_beras['Beras']).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = scaler.fit_transform(price_data.reshape(-1, 1))
    future_forecast,days = perform_forecasting_seasonal(selected_model, series, target_date_str)

    # Convert the result to a list before sending it as JSON
    result = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten().tolist()
    print(result[-1],data_beras['Beras'].tail(1).values[0])
    price = 100*(result[-1]-data_beras['Beras'].tail(1).values[0])/data_beras['Beras'].tail(1).values[0]
    if price < 0:
        color = "red"
    else:
        color = "blue"
    return jsonify({"estimated_revenue":round(price,3),'color':color})

@app.route('/predict_estimated_revenue', methods=['POST'])
def predict_estimated_revenue():
    # Get data from the request
    target_date_str = request.json['seasonal']
    total_available = request.json['total_available']
    
    # Perform future forecasting until the target date
    price_data = np.array(data_beras['Beras']).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    series = scaler.fit_transform(price_data.reshape(-1, 1))
    future_forecast,days = perform_forecasting_seasonal(selected_model, series, target_date_str)

    # Convert the result to a list before sending it as JSON
    result = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten().tolist()
    print(result[-1],data_beras['Beras'].tail(1).values[0])
    price = 100*(result[-1]-data_beras['Beras'].tail(1).values[0])/data_beras['Beras'].tail(1).values[0]
    new_price = total_available*price + total_available
    return jsonify({"estimated_revenue":new_price})

@app.route('/elnino_lanina', methods=['POST'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    start_date = request.args.get('start_date')
    username = 'fajar_fauzan'
    password = 'ajdMYk291R'

    # Fetch temperature data
    url_temperature = f"https://api.meteomatics.com/{start_date}T00:00:00Z--{start_date}T00:00:00Z:PT24H/t_mean_2m_24h:C/{lat},{lon}/html"
    response_temperature = requests.get(url_temperature, auth=(username, password))

    if response_temperature.status_code == 200:
        # Parse HTML content
        soup_temperature = BeautifulSoup(response_temperature.content, 'html.parser')
        csv_data_temperature = soup_temperature.find('pre', {'id': 'csv'}).text.strip()
        rows_temperature = csv_data_temperature.split('\n')[1:]  # Skip header
        temperature_value = float(rows_temperature[0].split(';')[1])
    else:
        print(f"Failed to fetch temperature data. Status code: {response_temperature.status_code}")
        temperature_value = None

    # Fetch precipitation data
    url_precipitation = f"https://api.meteomatics.com/{start_date}T00:00:00ZP100D:P1D/precip_24h:mm/{lat},{lon}/json"
    response_precipitation = requests.get(url_precipitation, auth=(username, password))

    if response_precipitation.status_code == 200:
        precipitation_data = response_precipitation.json()
        precipitation_value = precipitation_data['data'][0]['coordinates'][0]['dates'][0]['value']
    else:
        print(f"Failed to fetch precipitation data. Status code: {response_precipitation.status_code}")
        precipitation_value = None

    # Fetch wind speed data
    url_wind_speed = f"https://api.meteomatics.com/{start_date}T00:00:00ZP100D:P1D/wind_speed_10m:ms/{lat},{lon}/json"
    response_wind_speed = requests.get(url_wind_speed, auth=(username, password))

    if response_wind_speed.status_code == 200:
        wind_speed_data = response_wind_speed.json()
        wind_speed_value = wind_speed_data['data'][0]['coordinates'][0]['dates'][0]['value']
    else:
        print(f"Failed to fetch wind speed data. Status code: {response_wind_speed.status_code}")
        wind_speed_value = None

    new_data = pd.DataFrame({
        'temperature_2m_max (°C)': [temperature_value],
        'precipitation_sum (mm)': [precipitation_value],
        'wind_speed_10m_max (km/h)': [wind_speed_value]
    })
    catboost_proba = catboost_model.predict_proba(new_data)
    return jsonify({"El Nino":round(catboost_proba[0][0],0), "La Nina":round(catboost_proba[0][1],0)})

if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
