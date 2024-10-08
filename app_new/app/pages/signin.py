from dash import html, register_page, dcc
from dash_core_components import Store


register_page(__name__, path='/')
layout = html.Div([
    html.H2('Sign In', id='page-titles'),
    html.Div([
        # html.Div(className='shape'),
        # html.Div(className='shape'),
    ],className='background'),
    html.Form([
        
    
        html.Label('Username: Atleast 3 character long',className='username-title'),
        dcc.Input(id='user-input', type='text', placeholder='Enter your username',className='username-input', name='username', required=True, pattern=".{3,25}" ),
        Store(id='my-store', data='Initial Value'),

        html.Label('Password', className='password-title'),
        html.Div([html.Ul([html.Li("At least one number"),html.Li("One uppercase and lowercase letter"),html.Li("At least 8 or more characters")])], className='password-info'),
        dcc.Input(id='pass-input', type='password', placeholder='Enter your password',className='password-input', name='password', pattern="(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}", required=True),
        html.Button('Submit', id='submit-btn', n_clicks=0, className='submit-button'),
        # html.A(html.Button('Submit', id='submit-btn', n_clicks=0), href=dcc.Link('/graph')),
        # html.Button('clear', type='clear',className='clear-button'),
        dcc.Link('Dont have account? Sign up', href='/signup',className='signup-link'),
        html.Div(id='out-message'),
        
    ],className='form-signin', action='signin', method='post', n_clicks=0)
],className='body-signin',style={'background-image': 'url("/assets/background1.png")', 
                             'background-size': 'cover', 'background-repeat': 'no-repeat',
                             'background-position': 'center'
                             ,'width': '100%',
                             'height': '100%',
                             'z-index': '-2'})