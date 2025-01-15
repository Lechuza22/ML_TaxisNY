from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from dash import Dash, dcc, html, Input, Output, dash_table
from starlette.middleware.wsgi import WSGIMiddleware
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Configuración de FastAPI
    app = FastAPI()

    # Cargar datasets
    taxi_data_path = 'green_tripdata_2024-10_reducido.csv'
    yellow_taxi_path = 'Yellow_Tripdata_2024-10_reducido.csv'
    zone_data_path = 'transformed_taxi_zone_merged_with_locations.csv'

    taxi_data = pd.read_csv(taxi_data_path)
    yellow_data = pd.read_csv(yellow_taxi_path)
    zone_data = pd.read_csv(zone_data_path)

    # Validar datos cargados
    if taxi_data.empty or yellow_data.empty or zone_data.empty:
        raise ValueError("Uno o más datasets están vacíos. Verifica los archivos de entrada.")

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
    yellow_data = yellow_data.merge(zone_data, left_on='PULocationID', right_on='locationid_x', how='left')

    # Validar columnas requeridas
    required_columns = ['borough_x', 'PULocationID', 'fare_amount', 'trip_distance']
    for col in required_columns:
        if col not in data.columns or col not in yellow_data.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    # Filtrar datos válidos
    data = data[data['borough_x'].notna()]
    yellow_data = yellow_data[yellow_data['borough_x'].notna()]

    valid_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'EWR']
    data = data[data['borough_x'].isin(valid_boroughs)]
    yellow_data = yellow_data[yellow_data['borough_x'].isin(valid_boroughs)]

    data['zone_name'] = data['borough_x']
    yellow_data['zone_name'] = yellow_data['borough_x']

    data['pickup_hour'] = data['pickup_hour'].astype(int)
    yellow_data['pickup_hour'] = yellow_data['pickup_hour'].astype(int)

    # Función para eliminar valores atípicos
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

    data = remove_outliers(data, 'trip_distance')
    data = remove_outliers(data, 'fare_amount')
    yellow_data = remove_outliers(yellow_data, 'trip_distance')
    yellow_data = remove_outliers(yellow_data, 'fare_amount')

    # Función para calcular demanda semanal
    def calculate_weekly_demand(day_of_week, df):
        day_data = df[df['pickup_day'] == day_of_week]
        if day_data.empty:
            return pd.DataFrame(columns=['zone_name', 'ganancia_promedio', 'distancia_promedio', 'cantidad_viajes', 'hora_pico'])
        zone_summary = (
            day_data.groupby('zone_name')
            .agg({
                'fare_amount': 'mean',
                'trip_distance': 'mean',
                'zone_name': 'count',
                'pickup_hour': lambda x: x.value_counts().idxmax()
            })
            .rename(columns={
                'fare_amount': 'ganancia_promedio',
                'trip_distance': 'distancia_promedio',
                'zone_name': 'cantidad_viajes',
                'pickup_hour': 'hora_pico'
            })
            .reset_index()
        )
        return zone_summary.sort_values(by='cantidad_viajes', ascending=False)

    # Configuración de Dash
    dash_app = Dash(
        __name__,
        requests_pathname_prefix="/dashboard/"
    )
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
        if demand_data.empty:
            return px.bar(title="Sin datos disponibles"), px.bar(title="Sin datos disponibles")
        fig_demand = px.bar(
            demand_data, x='zone_name', y='cantidad_viajes',
            title=f"Demanda el {day}"
        )
        fig_earning = px.bar(
            demand_data, x='zone_name', y='ganancia_promedio',
            title=f"Ganancia Promedio el {day}"
        )
        return fig_demand, fig_earning

    @dash_app.callback(
        [Output('demand-chart-yellow', 'figure'), Output('avg-earning-chart-yellow', 'figure')],
        Input('day-dropdown-yellow', 'value')
    )
    def update_yellow_charts(day):
        demand_data = calculate_weekly_demand(day, yellow_data)
        if demand_data.empty:
            return px.bar(title="Sin datos disponibles"), px.bar(title="Sin datos disponibles")
        fig_demand = px.bar(
            demand_data, x='zone_name', y='cantidad_viajes',
            title=f"Demanda el {day}"
        )
        fig_earning = px.bar(
            demand_data, x='zone_name', y='ganancia_promedio',
            title=f"Ganancia Promedio el {day}"
        )
        return fig_demand, fig_earning

    # Montar Dash en FastAPI
    app.mount("/dashboard", WSGIMiddleware(dash_app.server))

    @app.get("/")
    def read_root():
        return HTMLResponse('<div style="background-color:#1E2B3A; color:#76EEC6; text-align:center; padding:20px; font-size:1.5em; font-family:Arial, sans-serif;">'
                            '<img src="/mnt/data/Logo.png" alt="TaxiCom2.0 Logo" style="width:150px; margin-bottom:15px;">'
                            '<br>Bienvenidos a <strong>TaxiCom2.0</strong></div>'
                            '<p style="text-align:center; font-family:Arial, sans-serif;">'
                            'Visita <a href="/dashboard" style="color:#76EEC6; text-decoration:none; font-weight:bold;">/dashboard</a></p>')

except Exception as e:
    logger.error(f"Error al inicializar la app: {e}")
