from dash import html, register_page, dcc

register_page(__name__)
layout = html.Div([
    html.H2('Sign In'),
    html.Form([
        html.Label('Username'),
        dcc.Input(id='user-input', type='text', placeholder='Enter your username'),
        html.Label('Password'),
        dcc.Input(id='pass-input', type='password', placeholder='Enter your password'),
        html.Button('Submit', id='submit-btn', n_clicks=0, ),
        # html.A(html.Button('Submit', id='submit-btn', n_clicks=0), href=dcc.Link('/graph')),
        html.Button('clear', type='clear'),
        html.Div(id='out-message')
    ], action='/post', method='post' )
])