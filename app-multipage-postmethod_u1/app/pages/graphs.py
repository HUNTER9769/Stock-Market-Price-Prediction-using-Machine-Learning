from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

register_page(__name__)

layout = html.Div([
    dbc.Row([dbc.Col(html.H1('Graph Page', style={'margin-left':'40px','margin-top': '30px','height':'100px'}), width=12)]),
    

        # dcc.Graph(
            # Add graph data and layout here
    # dcc.RadioItems(options=[ 'AAPL'], value='AAPL', id='controls'),
    # html.H1('{}',style={'color':'white'},id='mape-markdown'),
    # html.H1('{}',id='r2-markdown'),
     
    # dcc.Dropdown(
    # id='controls',
    # options=[{'label': 'Apple', 'value': 'AAPL'},
    #          {'label': 'Amazon', 'value': 'AMZN'},
    #          {'label': 'Tata Motors', 'value': 'TATAMOTORS.NS'}],
    # value='AAPL'
    # ),
    # dcc.Dropdown(
    # id='controls1',
    # options=[{'label': 'LSTM', 'value': 'LSTM'},
    #          {'label': 'Linear Regression', 'value': 'LR'}],
    # value='LR'
    # ),
    
    
    # dcc.Graph(id='second-graph'),
    dbc.Row(
            [
                dbc.Col(dcc.Dropdown(   id='controls',
                                        options=[{'label': 'Apple', 'value': 'AAPL'},
                                                {'label': 'Amazon', 'value': 'AMZN'},
                                                {'label': 'Tata Motors', 'value': 'TATAMOTORS.NS'}],
                                        value='AAPL'
                ), width=3, align="center", style={'height':'100px'}),
                dbc.Col(dcc.Dropdown(   id='controls1',
                                        options=[{'label': 'LSTM', 'value': 'LSTM'},
                                                {'label': 'Linear Regression', 'value': 'LR'}],
                                        value='LR'
                ), width=3, align="center", style={'height':'100px'}),
                # dbc.Col(html.Div("One of three columns"), width=3),
            ]
    , justify="center"),
    dbc.Row(
        dbc.Col(html.Div())
    ),

    dbc.Row(
        dbc.Col(html.Div(id='mape-markdown', children=[]), width={"size": 6, "offset": 1}, align='start')
    ),

    
    # html.Div(id='r2-markdown', children=[]),
    dbc.Row(
            [
                dbc.Col(dcc.Graph(id='first-graph'), width=6,align="center"),
                # dbc.Col(html.Div("One of three columns")),
                # dbc.Col(html.Div("One of three columns"), width=3),
            ]
    ),
    dbc.Row(
            [
                dbc.Col(dcc.Graph(id='second-graph', style={'width':'1400px', 'margin-left':'50px'}),align="center"),
                # dbc.Col(html.Div("One of three columns")),
                # dbc.Col(html.Div("One of three columns"), width=3),
            ]
    , justify="center"),
    # html.Div(id='hidden-div')
        # )
])