import sqlalchemy as sa
import plotly.graph_objects as go
from sqlalchemy import insert
from .server import db
from .models import gapminder, user
import flask

from dash import Dash, html, dcc, Input, Output, page_container, State
from dash.exceptions import PreventUpdate
import dash._pages


#model imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
        dcc.Link('SignUp', href='/signup', className='navbar'),
        dcc.Link('SignIn', href='/', className='navbar'),
        # dcc.Link('Graph', href='/graph'),
    ], className='navbar') 

    # App layout
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        # navbar,
        # html.Div(id='page-content'),
        # html.Div([
        # html.Div(
        #     dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        # ) for page in dash.page_registry.values()
        # ], className='navbar'),
        navbar,
        page_container
        # dcc.Location(id='url', refresh=False)
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

    # def page_forward(page):
    #     return html.Div([
    #         html.Div(
    #             dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
    #         ) for page in dash.page_registry.values()
    #     ])

    @app.callback(
        Output(component_id='first-graph', component_property='figure'),
        Input(component_id='controls', component_property='value')
    )
    # def update_graph(col_chosen):     
    #     query = db.session.query(
    #         gapminder.continent, 
    #         sa.func.avg(getattr(gapminder, col_chosen)).label('measure'))\
    #             .group_by(gapminder.continent).all()
        
    #     x = [q.continent for q in query]
    #     y = [q.measure for q in query]
    #     fig = go.Figure([go.Bar(x=x, y=y)])
    #     return fig
    def prediction_graph(self):
        def fetch_data(ticker, start_date, end_date):
            data = yf.download(ticker, start=start_date, end=end_date)
            data = data.drop(columns=['Volume', 'Adj Close'])
            return data
        
        def preprocess_data(data):
            # Select the 'Close' column for prediction
            data = data[['Close']]
            
            # Convert the data to numpy array
            dataset = data.values
            
            # Calculate the training data length
            training_data_len = int(np.ceil(len(dataset) * 0.8))
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            
            # Create the training data set
            train_data = scaled_data[0:int(training_data_len), :]
            
            # Create the x_train and y_train data sets
            x_train = []
            y_train = []
            
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])
            
            # Convert x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)
            
            # Reshape the data to 3D
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            return x_train, y_train, scaler, training_data_len, scaled_data, dataset
        
        def build_model():
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dense(units=25))
            model.add(Dense(units=1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            return model
        
        def train_model(model, x_train, y_train):
            # Train the model
            model.fit(x_train, y_train, batch_size=1, epochs=10)

        def make_predictions(model, scaler, scaled_data, training_data_len, dataset):
            # Create the testing data set
            test_data = scaled_data[training_data_len - 60:, :]
            
            # Create the x_test and y_test data sets
            x_test = []
            y_test = dataset[training_data_len:, :]
            
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])
            
            # Convert x_test to a numpy array
            x_test = np.array(x_test)
            
            # Reshape the data to 3D
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Get the models predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            return predictions, y_test
        
        def main(ticker, start_date, end_date):
            data = fetch_data(ticker, start_date, end_date)
            x_train, y_train, scaler, training_data_len, scaled_data, dataset = preprocess_data(data)
            model = build_model()
            train_model(model, x_train, y_train)
            predictions, y_test = make_predictions(model, scaler, scaled_data, training_data_len, dataset)
            # plot_predictions(data, training_data_len, predictions)

            # Plot the data
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            
            # fig = go.Figure()
            
            # fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
            # fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
            # fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
            
            
            # fig.update_layout(
            #     title='Model',
            #     xaxis_title='Date',
            #     yaxis_title='Close Price USD ($)',
            #     legend=dict(x=0, y=1, traceorder='normal')
            # )
            
            # f1 = go.FigureWidget(fig.show())
            layout = go.Layout(title='Model',
                xaxis_title='Date',
                yaxis_title='Close Price USD ($)',
                legend=dict(x=0, y=1, traceorder='normal'))

            fig = go.Figure(data=data, layout=layout)
            # return f1
            scatter1 = go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train')
            scatter2 = go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val')
            scatter3 = go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions')
            data = [scatter1, scatter2, scatter3]
            
            
            fig1 = go.FigureWidget(fig)

            return fig1
        
        main('AAPL', '2010-01-01', '2024-05-24')

    
    # @app.callback(
    #     Output(component_id='output-message', component_property='children'),
    #     Input(component_id='submit-button', component_property='n_clicks'),
    #     State('username-input', 'value'),
    #     State('email-input', 'value'),
    #     State('password-input', 'value')
    # )
    # def save_to_db(n_clicks, username, email, password):
        
    #     if n_clicks is not None:
    #     # save_to_database(username, email, password)
    #         # query = db.session.query(user).filter_by(username=username)
    #         db.session.add(user(index=db.session.query(user).count()+1, username=username, email=email, password=password))
    #         db.session.commit()
    #         # user(username=username, email=email, password=password).save()
    #         return html.Div('{} user has signed up successfully'.format(username))
        
    # @app.callback(
    # Output('url', 'pathname'),
    # Output('out-message', 'children'),
    # Input('submit-btn', 'n_clicks'),
    # State('user-input', 'value'),
    # State('pass-input', 'value')
    # )
    # def check_user_credentials(n_clicks, username, password):
    #     if n_clicks:
    #     # Add your logic to check if the user exists with the provided credentials
    #     # Assume you have a User model with a method to check user credentials
    #         if username and password:
    #             user_exists = bool(db.session.query(user).filter_by(username=username).scalar()) and bool(db.session.query(user).filter_by(password=password).first() ) # Replace with actual method to check credentials

    #             if user_exists:
    #                 return html.Div(f'Welcome, {username}! Redirecting to the graph page...')
    #                 # return flask.redirect('http://127.0.0.1:5000/graph')
    #                 # return html.Div(f'Welcome, {username}!')
    #                 # return page_forward(dash.page_registry["graph"])
    #                 # return dcc.Location(pathname="/graph", id="someid_doesnt_matter")
    #             else:
    #                 return PreventUpdate, html.Div('Invalid username or password. Please try again.')
    #         else:
    #             return PreventUpdate, html.Div('Please enter both username and password.')
    #     else:
    #         return PreventUpdate, None
        
    @app.server.route('/on_post', methods=['POST'])
    def on_post():
        data = flask.request.form
        username = data['username']
        email = data['email']
        password = data['password']
        user_exists = bool(db.session.query(user).filter_by(username=username).scalar()) and bool(db.session.query(user).filter_by(password=password).first() ) # Replace with actual method to check credentials
        if user_exists:
            return '{} user already exists. Try a different username or password'.format(username)
        else:
            db.session.add(user(index=db.session.query(user).count()+1, username=username, email=email, password=password))
            db.session.commit()
        # user(username=username, email=email, password=password).save()
            return '{} user has signed up successfully'.format(username)
        # return flask.redirect('/graph')
        
    @app.server.route('/signin', methods=['POST'])
    def signin():
        data = flask.request.form
        username = data['username']
        # email = data['email']
        password = data['password']
        user_exists = bool(db.session.query(user).filter_by(username=username).scalar()) and bool(db.session.query(user).filter_by(password=password).first() ) # Replace with actual method to check credentials
        if user_exists:
            return flask.redirect('/graph')
        else:
            # db.session.add(user(index=db.session.query(user).count()+1, username=username, email=email, password=password))
            # db.session.commit()
        # user(username=username, email=email, password=password).save()
            return 'Invalid username or password, or {} user does not exist. Try again'.format(username)
        # return flask.redirect('/graph')
    


    return app