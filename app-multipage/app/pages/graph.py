from dash import html, dcc, register_page

register_page(__name__)

layout = html.Div([
    
    html.H1('Graph Page'),

        # dcc.Graph(
            # Add graph data and layout here
    # dcc.RadioItems(options=[ 'AAPL'], value='AAPL', id='controls'),
    dcc.Dropdown(
    id='controls',
    options=[{'label': 'Apple', 'value': 'AAPL'},
             {'label': 'Tata Motors', 'value': 'TATAMOTORS.NS'}],
    value='AAPL'
    ),
    dcc.Dropdown(
    id='controls1',
    options=[{'label': 'LSTM', 'value': 'LSTM'}],
    value='LSTM'
    ),
    dcc.Graph(id='first-graph'),
    html.Div(id='hidden-div')
        # )
])