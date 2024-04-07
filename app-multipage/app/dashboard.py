import sqlalchemy as sa
import plotly.graph_objects as go
from sqlalchemy import insert
from .server import db
from .models import gapminder, user
import flask

from dash import Dash, html, dcc, Input, Output, page_container, State
import dash._pages


def create_dashapp(server):
    # Initialize the app
    app = Dash(__name__, server=server, use_pages=True )   

    # Define the layout for the sign-up page
    # sign_up_page = html.Div([
    #     html.H1('Sign Up Page'),
    #     # Add sign-up form components here
    # ])

    # # Define the layout for the sign-in page
    # sign_in_page = html.Div([
    #     html.H1('Sign In Page'),
    #     # Add sign-in form components here
    # ])

    # # Define the layout for the graph page
    # graph_page = html.Div([
    #     html.H1('Graph Page'),

    #     # dcc.Graph(
    #         # Add graph data and layout here
    #     dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls'),
    #     dcc.Graph(id='first-graph')
    #     # )
    # ])

    # Define the layout for the navbar
    navbar = html.Div([
        dcc.Link('Sign Up', href='/sign-up'),
        dcc.Link('Sign In', href='/sign-in'),
        dcc.Link('Graph', href='/graph'),
    ]) 

    # App layout
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        # navbar,
        # html.Div(id='page-content'),
        html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
        ]),
        page_container
        # html.Div(children='My First App with Dash & SqlAlchemy'),
        # html.Hr(),
        
    ])

    # @app.callback(
    # Output('page-content', 'children'),
    # Input('url', 'pathname')
    # )
    # def display_page(pathname):
    #     if pathname == '/sign-up' or pathname == '/':
    #         return sign_up_page
    #     elif pathname == '/sign-in':
    #         return sign_in_page
    #     elif pathname == '/graph':
    #         return graph_page
    #     else:
    #         return '404 Page Not Found'
        

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
    
    @app.callback(
        Output(component_id='output-message', component_property='children'),
        Input(component_id='submit-button', component_property='n_clicks'),
        State('username-input', 'value'),
        State('email-input', 'value'),
        State('password-input', 'value')
    )
    def save_to_db(n_clicks, username, email, password):
        
        if n_clicks is not None:
        # save_to_database(username, email, password)
            # query = db.session.query(user).filter_by(username=username)
            db.session.add(user(index=db.session.query(user).count()+1, username=username, email=email, password=password))
            db.session.commit()
            # user(username=username, email=email, password=password).save()
            return '{} user has signed up successfully'.format(username)
        
    @app.callback(
    Output('out-message', 'children'),
    Input('submit-btn', 'n_clicks'),
    Input('user-input', 'value'),
    Input('pass-input', 'value')
    )
    def check_user_credentials(n_clicks, username, password):
        user_exists = bool(db.session.query(user).filter_by(username=username).scalar()) and bool(db.session.query(user).filter_by(password=password).first() ) # Replace with actual method to check credentials
        # Add your logic to check if the user exists with the provided credentials
        # Assume you have a User model with a method to check user credentials
        if username and password:
            
            if user_exists:
                return html.Div(f'Welcome, {username}!')
                return user_exists
                # return dcc.Location(pathname="/graph", id="someid_doesnt_matter")
            else:
                return html.Div('Invalid username or password. Please try again.')
        else:
            return html.Div('Please enter both username and password.')

        
    @app.server.route('/post', methods=['POST'])
    def on_post():
        data = flask.request.form
        print(data)
        return flask.redirect('/graph')
    


    return app