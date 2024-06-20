from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

register_page(__name__)

layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # dcc.Store(id='username-store'),
    html.Div([
            html.H2("🧭"),
            html.Br(),
            dcc.Link('📶', href='/graphs',title='Dashboard', style={'text-decoration': 'none', 'font-size': '32px'}),
            html.Br(),
            dcc.Link('🔑', href='/', title='Logout', style={'text-decoration': 'none', 'font-size': '32px'}),
            html.Br(),
            dcc.Link('🔐', href='/signup',title='SignUp', style={'text-decoration': 'none', 'font-size': '32px'}),
            html.Br(),
            dcc.Link('🪪', href='/user',title='User Data', style={'text-decoration': 'none', 'font-size': '32px'}),
            # html.Ul([
            #     html.Li(dcc.Link('📶', href='/graphs')),
            #     html.Li(dcc.Link('🔑', href='/signin')),
            #     html.Li(dcc.Link('🔐', href='/signup')),
            #     html.Li(dcc.Link('🪪', href='/user')),
            # ], style={'list-style-type': 'none','text-decoration': 'none', 'padding': 0}),
        ], style={'padding': '10px', 'background-color': '#f0f0f0', 'width': '5%', 'position': 'fixed', 'height': '100%'}),
    
    

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
    
    html.Div( [
    # dcc.Graph(id='second-graph'),

    dbc.Row([dbc.Col(html.H1('Graph Page'), id='page-titles',  width=12)]),
    html.Div(id='user-cred'),
    # dbc.Row([dbc.Col(html.H1('Graph Page', style={'margin-left':'40px','margin-top': '30px','height':'100px'}), width=12)]),
    dbc.Row(
            [
                dbc.Col(dcc.Dropdown(   id='controls',
                                        options=[{'label': 'Apple', 'value': 'AAPL'},
                                                {'label': 'Amazon', 'value': 'AMZN'},
                                                {'label': 'Tata Motors', 'value': 'TATAMOTORS.NS'}],
                                        value='AAPL',
                style={'color':'black'}), width=3, align="left", style={'color':'black','height':'50px'}),
                dbc.Col(dcc.Dropdown(   id='controls1',
                                        options=[{'label': 'LSTM', 'value': 'LSTM'},
                                                {'label': 'Linear Regression', 'value': 'LR'},
                                                {'label': 'Decision Tree', 'value': 'DT'},
                                                {'label': 'Random Forest', 'value': 'RF'},
                                                {'label': 'SVM', 'value': 'SVM'},],
                                        value='LR'
                ), width=3, align="left", style={'color':'black','height':'50px'}),
                # dbc.Col(html.Div("One of three columns"), width=3),
            ]
    , justify="center"),
    

    dbc.Row(
        dbc.Col(html.Div(id='mape-markdown', children=[]), width={"size": 6}, align='start', style={'margin':'20px'})
    ),

    
    # html.Div(id='r2-markdown', children=[]),
    dbc.Row(
            [
                dbc.Col(dcc.Graph(id='first-graph', style={'width':'1420px'} ), width='auto',align="center"),
                # dbc.Col(html.Div("One of three columns")),
                # dbc.Col(html.Div("One of three columns"), width=3),
            ]
    ),
    dbc.Row(
            [
                dbc.Col(dcc.Graph(id='second-graph', style={'width':'1420px'}),align="center"),
                # dbc.Col(html.Div("One of three columns")),
                # dbc.Col(html.Div("One of three columns"), width=3),
            ]
    , justify="center"),
    # html.Div(id='hidden-div')

    dbc.Row([
                dbc.Col(html.Div(id='table-container', style={'width':'600px', 'height':'800px'}) , width='auto' , align="left", style={'margin-left':'10px'}),

                dbc.Col(html.Div([dcc.Dropdown(   id='metric-dropdown',
                                        options=[{'label': '1d', 'value': '1d'},
                                                {'label': '5d', 'value': '5d'},
                                                {'label': '1mo', 'value': '1mo'},
                                                {'label': '3mo', 'value': '3mo'},
                                                {'label': '6mo', 'value': '6mo'},
                                                {'label': '1y', 'value': '1y'},
                                                {'label': '2y', 'value': '2y'},
                                                {'label': '5y', 'value': '5y'},
                                                {'label': '10y', 'value': '10y'},
                                                {'label': 'ytd', 'value': 'ytd'},
                                                {'label': 'max', 'value': 'max'},
                                                ],
                                        value='1d', style={'color':'black'}), 
                                        
                        html.Div(id='metric-div', children=[]),                
                        dcc.Graph(id='metric-graph', style={'width':'746px'})
                        
                        ]),id='metric-container',width='auto' ,align="right")
            ]
    )
        # )
    ],id='page-content',style={'margin-left': '5%'} )
])
