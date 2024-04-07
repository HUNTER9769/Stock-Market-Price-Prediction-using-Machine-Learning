from dash import html, register_page, dcc

register_page(__name__, path='/')

layout = html.Div([
    html.H1('Sign Up Page'),
    html.Label('Username:'),
    dcc.Input(id='username-input', type='text', value=''),
    html.Label('Email:'),
    dcc.Input(id='email-input', type='text', value=''),
    html.Label('Password:'),
    dcc.Input(id='password-input', type='password', value=''),
    html.Button('Submit', id='submit-button'),
    html.Div(id='output-message')
])