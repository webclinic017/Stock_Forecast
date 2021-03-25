import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output



import sys
app_path = r'/Users/Aron/Documents/GitHub/Data/Stock_Analysis/Dev/Multi-Page/Method2/apps'
path_codebase = [r'/Users/Aron/Documents/GitHub/Data/Stock_Analysis/Dev/Multi-Page/Method2',
                 app_path]



for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path




from app import app

layout = html.Div([
    html.H3('App 1'),
    dcc.Dropdown(
        id='app-1-dropdown',
        options=[
            {'label': 'App 1 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to App 2', href='http://127.0.0.1:8050/apps/app2'),
    dcc.Link('Go to Index', href='http://127.0.0.1:8050/'),    
])


@app.callback(
    Output('app-1-display-value', 'children'),
    Input('app-1-dropdown', 'value'))


def display_value(value):
    return 'You have selected "{}"'.format(value)