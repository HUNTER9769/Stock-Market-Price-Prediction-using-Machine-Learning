from dash import html, register_page, dcc

register_page(__name__)

layout = html.Div([
    html.H1('Sign Up Page', id='page-titles'),
    html.Div([
        # html.Div(style={'margin-left':'530px'},className='shape', ),
        # html.Div(style={'margin-right':'530px'},className='shape'),
    ],className='background'),
    html.Form([
        html.Label('Username: Atleast 3 character long'),
        dcc.Input(id='username-input', type='text', name='username', placeholder='Enter your username', required=True, pattern=".{3,25}" ),
        html.Label('Email:'),
        dcc.Input(id='email-input', type='email', name='email', placeholder='Enter your email', required=True),
        html.Label('Password:'),
        html.Div([html.Ul([html.Li("At least one number"),html.Li("One uppercase and lowercase letter"),html.Li("At least 8 or more characters")])], className='password-info'),
        dcc.Input(id='password-input', type='password', name='password', pattern="(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}", required=True, placeholder='Enter your password'),
        html.Button('Submit', id='submit-button'),
        html.Div(id='output-message')
        
    ],className='form-signup', method='post', n_clicks=0, action='/on_post')
],className='body-signup',style={'background-image': 'url("/assets/background1.png")', 
                             'background-size': 'cover', 'background-repeat': 'no-repeat',
                             'background-position': 'center',
                             'width': '100%',
                             'height': '100%',
                             'z-index': '-2'})