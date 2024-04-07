from dash import html, dcc, register_page

register_page(__name__)

layout = html.Div([
    
    html.H1('Graph Page'),

        # dcc.Graph(
            # Add graph data and layout here
    dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls'),
    dcc.Graph(id='first-graph'),
    html.Div(id='hidden-div')
        # )
])