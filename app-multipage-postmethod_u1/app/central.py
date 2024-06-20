import sqlalchemy as sa
import plotly.graph_objects as go
from sqlalchemy import insert
from .server import db
from .models import gapminder, user
import flask

from dash import Dash, html, dcc, Input, Output, page_container, State
from dash.exceptions import PreventUpdate
import dash._pages
import dash_bootstrap_components as dbc

#model imports
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split






def create_dashapp(server):
    # Initialize the app
    global app
    app = Dash(__name__, server=server, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP]) 
    
    

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
    # navbar = html.Div([
    #     dcc.Link('SignUp', href='/signup', className='navbar'),
    #     dcc.Link('SignIn', href='/', className='navbar'),
    #     # dcc.Link('Graph', href='/graph'),
    # ], className='navbar') 


    navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("SignUp", href="/signup")),
        dbc.NavItem(dbc.NavLink("SignIn", href="/")),
    ],
    brand="",
    brand_href="#",
    color="#004A77",
    # dark=True,
    className='navbar',
    dark=True
    )
    # App layout
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        # html.Div(id='page-content'),
        # html.Div([
        # html.Div(
        #     dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        # ) for page in dash.page_registry.values()
        # ], className='navbar'),
        navbar,
        page_container

        
    ])

    # @app.callback(
    # Output('page-content', 'children'),
    # Input('url', 'pathname')
    # )
    # def display_page(pathname):
    #     if pathname == '/sign-in' or pathname == '/':
    #         return sign_in_page
    #     elif pathname == '/sign-up':
    #         return sign_up_page
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

    # @app.callback(
    #     Output(component_id='first-graph', component_property='figure'),
    #     Input(component_id='dropdown', component_property='value')
    # )
    # def update_graph(col_chosen):     
    #     query = db.session.query(
    #         gapminder.continent, 
    #         sa.func.avg(getattr(gapminder, col_chosen)).label('measure'))\
    #             .group_by(gapminder.continent).all()
        
    #     x = [q.continent for q in query]
    #     y = [q.measure for q in query]
    #     fig = go.Figure([go.Bar(x=x, y=y)])
    #     return fig
    

    
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
    # from app.pages import graphs
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
            return flask.redirect('/graphs')
        else:
            # db.session.add(user(index=db.session.query(user).count()+1, username=username, email=email, password=password))
            # db.session.commit()
        # user(username=username, email=email, password=password).save()
            return 'Invalid username or password, or {} user does not exist. Try again'.format(username)
        # return flask.redirect('/graph')


    
    @app.callback(
        Output(component_id='first-graph', component_property='figure'),
        Output(component_id='second-graph', component_property='figure'),
        Output(component_id='mape-markdown', component_property='children'),
        # Output(component_id='r2-markdown', component_property='children'),
        Input(component_id='controls', component_property='value'),
        Input(component_id='controls1',component_property='value'),
        # State(component_id='controls', component_property='value'),
        # State(component_id='controls1',component_property='value'),

    )
    def update_graph( company, select_model):     
        def model_selection(select_model, data, future_date):
            if select_model == 'LR':
                data = data[['Close']]
                dataset = data.values
                training_data_len = int(np.ceil(len(dataset) * 0.8))
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)

                # Prepare training data
                train_data = scaled_data[:training_data_len, :]
                x_train, y_train = [], []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)

                # Train linear regression model
                model = LinearRegression()
                model.fit(x_train, y_train)

                # Prepare test data
                test_data = scaled_data[training_data_len - 60:, :]
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)

                # Make predictions
                predictions = model.predict(x_test)
                predictions = predictions.reshape(-1, 1)
                predictions = scaler.inverse_transform(predictions)

                mape = mean_absolute_percentage_error(y_test,predictions)
                r2 = r2_score(y_test, predictions)


                # Predict future date
                last_sequence = dataset[-60:]
                last_sequence_scaled = scaler.transform(last_sequence)
                x_input = np.reshape(last_sequence_scaled, (1, 60))
                future_prediction_scaled = model.predict(x_input)
                future_prediction = scaler.inverse_transform(future_prediction_scaled.reshape(-1, 1))

                # Plot results
                train = data[:training_data_len]
                valid = data[training_data_len:]
                valid['Predictions'] = predictions

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
                fig.add_trace(go.Scatter(x=[future_date], y=[future_prediction[0, 0]], mode='markers', name='Future Prediction', marker=dict(color='LightSkyBlue', size=10)))
                r2 = r2_score(y_test, predictions)
                final_pred_price = f"Predicted closing price for {company} on {future_date.strftime('%Y-%m-%d')} is: ₹{future_prediction[0, 0]:.2f} with {r2*100:.2f}% accuracy"
                fig.update_layout(
                title=final_pred_price,
                xaxis_title='Date',
                yaxis_title='Close Price (INR)' if '.NS' in company else 'Close Price (USD)',
                legend=dict(x=0, y=1, traceorder='normal'),
                template='plotly_dark'
                )
                # final_pred_price = f"Predicted closing price for {company} on {future_date.strftime('%Y-%m-%d')} is: ₹{future_prediction[0, 0]:.2f}"
                return fig, mape, r2, final_pred_price
            

            elif select_model == 'LSTM':

                data = data[['Close']]
                dataset = data.values
                training_data_len = int(np.ceil(len(dataset) * 0.8))
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(dataset)
                train_data = scaled_data[0:int(training_data_len), :]
                x_train = []
                y_train = []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                    


                model.fit(x_train, y_train, batch_size=1, epochs=5)


                test_data = scaled_data[training_data_len - 60:, :]
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)

                data = data[['Close']]
                dataset = data.values
                
                last_sequence = dataset[-60:]
                last_sequence_scaled = scaler.transform(last_sequence)
                x_input = np.reshape(last_sequence_scaled, (1, last_sequence_scaled.shape[0], 1))
                future_prediction_scaled = model.predict(x_input)
                future_prediction = scaler.inverse_transform(future_prediction_scaled)

                train = data[:training_data_len]
                valid = data[training_data_len:]
                valid['Predictions'] = predictions
                fig = go.Figure()
                future_price = future_prediction[0, 0]
                fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Val'))
                fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions'))
                fig.add_trace(go.Scatter(x=[future_date], y=[future_prediction[0, 0]], mode='markers', name='Future Prediction',marker=dict(color='LightSkyBlue', size=10)))
                r2 = r2_score(y_test, predictions)
                final_pred_price = f"Predicted closing price for {company} on {future_date.strftime('%Y-%m-%d')} is: ₹{future_prediction[0, 0]:.2f} with {r2*100:.2f}% accuracy"
                
                fig.update_layout(
                    title=final_pred_price,
                    xaxis_title='Date',
                    yaxis_title='Close Price (INR)' if '.NS' in company else 'Close Price (USD)',
                    legend=dict(x=0, y=1, traceorder='normal'),
                    template='plotly_dark'
                )
                # final_pred_price = f"Predicted closing price for {company} on {future_date.strftime('%Y-%m-%d')} is: ₹{future_prediction[0, 0]:.2f}"
                    
                mape = mean_absolute_percentage_error(y_test, predictions)
                
                return fig, mape, r2, final_pred_price


        def main(ticker, start_date, end_date, future_date):
            data = yf.download(ticker, start=start_date, end=end_date)
            # Linearregression(data,future_date)
            fig , mape, r2, final_pred_price = model_selection(select_model, data, future_date)
            #x_train, y_train, scaler, training_data_len, scaled_data, dataset = preprocess_data(data)
            #model = train_model(x_train, y_train)
            #predictions, y_test = make_predictions(model, scaler, scaled_data, training_data_len, dataset)
            #future_date, future_price = predict_future_date(model, scaler, data, future_date)
            #plot_predictions(data, training_data_len, predictions, future_date, future_price)
            #print(f"Predicted closing price for {ticker} on {future_date.strftime('%Y-%m-%d')} is: ₹{future_price:.2f}")

            # mape, r2 = calculate_accuracy(y_test, predictions)
            # print("Mean Absolute Percentage Error (MAPE):", mape)
            # print("R-squared (R2) Score:", r2)

            return fig, mape, r2

        
            

        def main2(ticker, start_date, end_date):
            data = yf.download(ticker, start=start_date, end=end_date)
            fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])
            fig.update_layout(
                    title='Stock Price Candle Stick Chart',
                    # title_text = 'Stock Price Candle Stick Chart',
                    xaxis_title='Date',
                    yaxis_title='Close Price INR (₹)',
                    # legend=dict(x=0, y=1, traceorder='normal'),
                    template='plotly_dark'
                )
            return fig
        fig, mape, r2 = main(company, '2021-01-01', '2024-05-29', pd.Timestamp('2024-05-29'))
        fig2 = main2(company, '2021-01-01', '2024-05-29')
        return fig, fig2, html.Div([
            html.P('Mean Absolute Percentage Error (MAPE): {:.2f}'.format(mape)),
            html.P('R -squared (R2) Score: {:.2f}'.format(r2)),
            ])
        # , html.Div([
        #     html.H3('{}'.format(final_pred_price)),   
        # ])
    
    # @app.callback(
    # Output('output', 'children'),
    # [Input('submit-button', 'n_clicks')],
    # [State('company-dropdown', 'value'),
    #  State('model-dropdown', 'value')]
    # )
    # def predict_stock_price(n_clicks, company, model):
    #     if n_clicks > 0:
    #         # # Load the selected company's stock data
    #         # company_data = stock_data[stock_data['Company'] == company]
            
    #         # # Preprocess the data
    #         # scaler = MinMaxScaler()
    #         # scaled_data = scaler.fit_transform(company_data['Price'].values.reshape(-1, 1))
            
    #         # # Train the selected model
    #         # models[model].fit(scaled_data[:-1], scaled_data[1:])
            
    #         # # Make a prediction
    #         # next_price = models[model].predict(scaled_data[-1:].reshape(1, 1))[0][0]
            
    #         # # Scale the prediction back to the original range
    #         # original_price = scaler.inverse_transform([[next_price]])[0][0]
            
    #         # Redirect to the graph page
    #         # return html.Div([
    #         #     html.H2(f'Predicted stock price for {company}: ${original_price:.2f}'),
    #         #     html.A('View Graph', href='/graph')
    #         # ])
    #         return flask.redirect('/graph')
    #     else:
    #         return ''

    return app