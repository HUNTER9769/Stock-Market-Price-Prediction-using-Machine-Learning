from dash import html, register_page, dcc

register_page(__name__)
layout = html.Div([
    html.H2('Sign In'),
    html.Div([
        html.Div(className='shape'),
        html.Div(className='shape'),
    ],className='background'),
    html.Form([
        
    
        html.Label('Username',className='username-title'),
        dcc.Input(id='user-input', type='text', placeholder='Enter your username',className='username-input'),
        html.Label('Password', className='password-title'),
        dcc.Input(id='pass-input', type='password', placeholder='Enter your password',className='password-input'),
        html.Button('Submit', id='submit-btn', n_clicks=0, className='submit-button'),
        # html.A(html.Button('Submit', id='submit-btn', n_clicks=0), href=dcc.Link('/graph')),
        html.Button('clear', type='clear',className='clear-button'),
        dcc.Link('Dont have account? Sign up', href='/',className='signup-link'),
        html.Div(id='out-message'),
        
    ],className='form')
],className='body')