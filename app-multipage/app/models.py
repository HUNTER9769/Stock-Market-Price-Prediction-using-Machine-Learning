from .server import db
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class gapminder(Base):
    __tablename__ = "gapminder2007"

    id = db.Column(db.Integer, primary_key=True)
    country = db.Column(db.String)
    pop = db.Column(db.Integer)
    continent = db.Column(db.String)
    lifeExp = db.Column(db.Float)
    gdpPercap = db.Column(db.Float)


class user(Base):
    __tablename__ = "user"
    index = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)