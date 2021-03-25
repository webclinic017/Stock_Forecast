from dash.dependencies import Input, Output




import sys
path_codebase = [r'/Users/Aron/Documents/GitHub/Data/Stock_Analysis/Dev/Multi-Page/Method2',
                 r'/Users/Aron/Documents/GitHub/Data/Stock_Analysis/Dev/Multi-Page/Method2/apps']



for i in path_codebase:    
    if i not in sys.path:
        sys.path = [i] + sys.path

# from app import app

# @app.callback(
#     Output('app-1-display-value', 'children'),
#     Input('app-1-dropdown', 'value'))
# def display_value(value):
#     return 'You have selected "{}"'.format(value)

# @app.callback(
#     Output('app-2-display-value', 'children'),
#     Input('app-2-dropdown', 'value'))
# def display_value(value):
#     return 'You have selected "{}"'.format(value)