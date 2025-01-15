from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from dash import Dash, dcc, html, Input, Output, dash_table
from starlette.middleware.wsgi import WSGIMiddleware

# Configuración de FastAPI
app = FastAPI()

# Cargar datasets
taxi_data_path = 'green_tripdata_2024-10_reducido.csv'
yellow_taxi_path = 'Yellow_Tripdata_2024-10_reducido.csv'
zone_data_path = 'transformed_taxi_zone_merged_with_locations.csv'

taxi_data = pd.read_csv(taxi_data_path)
yellow_data = pd.read_csv(yellow_taxi_path)
zone_data = pd.read_csv(zone_data_path)

# Preprocesar datos
taxi_data['lpep_pickup_datetime'] = pd.to_datetime(taxi_data['lpep_pickup_datetime'], errors='coerce')
yellow_data['tpep_pickup_datetime'] = pd.to_datetime(yellow_data['tpep_pickup_datetime'], errors='coerce')

taxi_data = taxi_data.dropna(subset=['lpep_pickup_datetime'])
yellow_data = yellow_data.dropna(subset=['tpep_pickup_datetime'])

taxi_data['pickup_hour'] = taxi_data['lpep_pickup_datetime'].dt.hour
taxi_data['pickup_day'] = taxi_data['lpep_pickup_datetime'].dt.day_name().map({'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'})

yellow_data['pickup_hour'] = yellow_data['tpep_pickup_datetime'].dt.hour
yellow_data['pickup_day'] = yellow_data['tpep_pickup_datetime'].dt.day_name().map({'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'})

data = taxi_data.merge(zone_data, left_on='PULocationID', right_on='locationid_x', how='left')
data = data[data['borough_x'].notna()]
data = data[data['borough_x'].isin(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR'])]
data['zone_name'] = data['borough_x']
data['pickup_hour'] = data['pickup_hour'].astype(int)
data['pickup_day'] = data['pickup_day']

yellow_data = yellow_data.merge(zone_data, left_on='PULocationID', right_on='locationid_x', how='left')
yellow_data = yellow_data[yellow_data['borough_x'].notna()]
yellow_data = yellow_data[yellow_data['borough_x'].isin(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR'])]
yellow_data['zone_name'] = yellow_data['borough_x']
yellow_data['pickup_hour'] = yellow_data['pickup_hour'].astype(int)
yellow_data['pickup_day'] = yellow_data['pickup_day']

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

data = remove_outliers(data, 'trip_distance')
data = remove_outliers(data, 'fare_amount')
yellow_data = remove_outliers(yellow_data, 'trip_distance')
yellow_data = remove_outliers(yellow_data, 'fare_amount')

def calculate_weekly_demand(day_of_week, df):
    day_data = df[df['pickup_day'] == day_of_week]
    zone_summary = (
        day_data.groupby('zone_name')
        .agg({
            'fare_amount': 'mean',
            'trip_distance': 'mean',
            'zone_name': 'count',
            'pickup_hour': lambda x: x.value_counts().idxmax()
        })
        .rename(columns={
            'fare_amount': 'avg_earning',
            'trip_distance': 'avg_distance',
            'zone_name': 'trip_count',
            'pickup_hour': 'peak_hour'
        })
        .reset_index()
    )
    return zone_summary.sort_values(by='trip_count', ascending=False)

def calculate_heatmap_data(df):
    heatmap_data = (
        df.groupby(['pickup_day', 'pickup_hour'])
        .size()
        .reset_index(name='trip_count')
    )
    hours = pd.DataFrame({'pickup_hour': range(24)})
    days = pd.DataFrame({'pickup_day': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']})
    full_index = hours.merge(days, how='cross')
    heatmap_data = full_index.merge(heatmap_data, on=['pickup_hour', 'pickup_day'], how='left').fillna(0)
    return heatmap_data

def predict_best_time_and_route(zone, df, day):
    zone_data = df[(df['zone_name'] == zone) & (df['pickup_day'] == day)]
    features = zone_data[['pickup_hour', 'trip_distance', 'passenger_count']]
    target = zone_data['fare_amount']

    if len(features) > 0 and len(target) > 0:
        features = pd.get_dummies(features, columns=['pickup_hour'], drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        best_row = X_test.iloc[np.argmax(predictions)]

        best_time = [int(col.split('_')[-1]) for col in best_row.index if col.startswith('pickup_hour_') and best_row[col]][0]
        avg_distance = zone_data['trip_distance'].mean()
        avg_fare = zone_data['fare_amount'].mean()

        return best_time, avg_distance, avg_fare

    return None, None, None

# Configuración de Dash
dash_app = Dash(__name__)
dash_app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Taxis Verdes', children=[
            html.H1("Demanda de Taxis Verdes"),
            dcc.Dropdown(
                id='day-dropdown-green',
                options=[{'label': day, 'value': day} for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']],
                value='Lunes'
            ),
            dcc.Graph(id='demand-chart-green'),
            dcc.Graph(id='avg-earning-chart-green'),
        ]),
        dcc.Tab(label='Taxis Amarillos', children=[
            html.H1("Demanda de Taxis Amarillos"),
            dcc.Dropdown(
                id='day-dropdown-yellow',
                options=[{'label': day, 'value': day} for day in ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']],
                value='Lunes'
            ),
            dcc.Graph(id='demand-chart-yellow'),
            dcc.Graph(id='avg-earning-chart-yellow'),
        ])
    ])
])

# Callbacks de Dash
@dash_app.callback(
    [Output('demand-chart-green', 'figure'), Output('avg-earning-chart-green', 'figure')],
    Input('day-dropdown-green', 'value')
)
def update_green_charts(day):
    demand_data = calculate_weekly_demand(day, data)
    fig_demand = px.bar(
        demand_data, x='zone_name', y='trip_count',
        title=f"Demanda el {day}", color='avg_earning'
    )
    fig_earning = px.bar(
        demand_data, x='zone_name', y='avg_earning',
        title=f"Ganancia Promedio el {day}"
    )
    return fig_demand, fig_earning

# Montar Dash en FastAPI
app.mount("/dashboard", WSGIMiddleware(dash_app.server))

# Ruta de FastAPI para la raíz
@app.get("/")
def read_root():
    return HTMLResponse("<h1>Bienvenido a la API</h1><p>Visita <a href='/dashboard'>/dashboard</a></p>")
