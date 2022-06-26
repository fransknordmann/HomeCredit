# Run this app with `python P7_03_app_test.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import json
import requests
from datetime import datetime

app = Dash(__name__)

colors = {
    'background': '#EEEEEE',
    'text': '#223322'
}


# read weather forecast from our API
METEO_API_MAISON = f'http://127.0.0.1:5000/api/meteo/'
response = requests.get(METEO_API_MAISON)
content_meteo = json.loads(response.content.decode('utf-8'))

data_meteo = []
for prev in content_meteo["data"]:
    datetimest = prev[0]
    date = datetime.fromtimestamp(datetimest//1000)
    temperature = prev[1]
    data_meteo.append([date, datetimest, temperature])

df_meteo = pd.DataFrame(data_meteo, columns=['date', 'datetimest', 'temp'])
graph_meteo = px.line(df_meteo, x='date', y='temp', )
graph_meteo.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Hotel": ["Scandic", "Thon", "Radisson", "Scandic", "Thon", "Radisson"],
    "Number": [1, 4, 1, 2, 3, 2],
    "City": ["Trondheim", "Trondheim", "Trondheim", "Bergen", "Bergen", "Bergen"]
})

fig = px.bar(df, x="Hotel", y="Number", color="City", barmode="group")

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Demo Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.H2(children='Plotly px.bar - manual dataframe - 2D bar', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-1',
        figure=fig
    ),

    html.H2(children='Plotly px.line - OpenWeather API - temperatures in Bodø', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=graph_meteo
    ),


])

if __name__ == '__main__':
    app.run_server(debug=True)