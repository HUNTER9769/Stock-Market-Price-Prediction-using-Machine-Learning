import sqlalchemy as sa
import plotly.graph_objects as go

from .server import db
from .models import gapminder
from dash import Dash, html, dcc, Input, Output


def create_dashapp(server):
    # Initialize the app
    app = Dash(__name__, server=server)   

    # Define the layout for the sign-up page
    sign_up_page = html.Div([
        html.H1('Sign Up Page'),
        # Add sign-up form components here
    ])

    # Define the layout for the sign-in page
    sign_in_page = html.Div([
        html.H1('Sign In Page'),
        # Add sign-in form components here
    ])

    # Define the layout for the graph page
    graph_page = html.Div([
        html.H1('Graph Page'),

        # dcc.Graph(
            # Add graph data and layout here
        dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls'),
        dcc.Graph(id='first-graph')
        # )
    ])

    # Define the layout for the navbar
    navbar = html.Div([
        dcc.Link('Sign Up', href='/sign-up'),
        dcc.Link('Sign In', href='/sign-in'),
        dcc.Link('Graph', href='/graph'),
    ]) 

    # App layout
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        navbar,
        html.Div(id='page-content')
        # html.Div(children='My First App with Dash & SqlAlchemy'),
        # html.Hr(),
        
    ])

    @app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
    )
    def display_page(pathname):
        if pathname == '/sign-up' or pathname == '/':
            return sign_up_page
        elif pathname == '/sign-in':
            return sign_in_page
        elif pathname == '/graph':
            return graph_page
        else:
            return '404 Page Not Found'
        

    @app.callback(
        Output(component_id='first-graph', component_property='figure'),
        Input(component_id='controls', component_property='value')
    )
    def update_graph(col_chosen):     
        query = db.session.query(
            gapminder.continent, 
            sa.func.avg(getattr(gapminder, col_chosen)).label('measure'))\
                .group_by(gapminder.continent).all()
        
        x = [q.continent for q in query]
        y = [q.measure for q in query]
        fig = go.Figure([go.Bar(x=x, y=y)])
        return fig

    return app