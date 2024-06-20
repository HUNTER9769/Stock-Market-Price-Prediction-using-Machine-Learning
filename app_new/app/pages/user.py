from dash import html, register_page, dcc
import dash_bootstrap_components as dbc

register_page(__name__, path='/user')

layout = html.Div([ 

    html.Div([
            html.H2("🧭"),
            html.Br(),
            dcc.Link('📶', href='/graphs', style={'text-decoration': 'none', 'font-size': '32px'}),
            html.Br(),
            dcc.Link('🔑', href='/', style={'text-decoration': 'none', 'font-size': '32px'}),
            html.Br(),
            dcc.Link('🔐', href='/signup', style={'text-decoration': 'none', 'font-size': '32px'}),
            html.Br(),
            dcc.Link('🪪', href='/user', style={'text-decoration': 'none', 'font-size': '32px'}),
            # html.Ul([
            #     html.Li(dcc.Link('📶', href='/graphs')),
            #     html.Li(dcc.Link('🔑', href='/signin')),
            #     html.Li(dcc.Link('🔐', href='/signup')),
            #     html.Li(dcc.Link('🪪', href='/user')),
            # ], style={'list-style-type': 'none','text-decoration': 'none', 'padding': 0}),
        ], style={'padding': '10px', 'background-color': '#f0f0f0', 'width': '5%', 'position': 'fixed', 'height': '100%'}),


        html.Div(id='user-div', children=[], style={'margin-left': '5%'} )
])