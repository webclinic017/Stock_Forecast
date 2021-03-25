


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
from apps import app1, app2


app.layout = html.Div([
    # dcc.Input(id='url'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.P('Index'),
        dcc.Link('Go to App1', href='http://127.0.0.1:8050/apps/app1'),    
            dcc.Link('Go to App2', href='http://127.0.0.1:8050/apps/app2'),    
    
    
    print('test')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))

def display_page(pathname):

    print('Callback')
    print(pathname)
    
    if pathname == '/apps/app1':
        print('app1')
        return app1.layout
    elif pathname == '/apps/app2':
        return app2.layout
    else:
        print('404')
        return '404'

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server()    