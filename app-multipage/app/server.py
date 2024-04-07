from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

class Config(object):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

db = SQLAlchemy()

def create_app():
    server = Flask(__name__)
    server.config.from_object(Config)
    db.init_app(server)
    
    with server.app_context():
        df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
        df.to_sql('gapminder2007', con=db.engine, if_exists='replace')
        df1 = pd.read_csv('../app-multipage/user.csv')
        df1.to_sql('user', con=db.engine, if_exists='replace')

    from .dashboard import create_dashapp
    dash_app = create_dashapp(server)
    return server # or dash app if you use debug mode in dash