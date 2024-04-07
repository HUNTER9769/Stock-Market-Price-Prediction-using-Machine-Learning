from .server import db


class gapminder(db.Model):
    __tablename__ = "gapminder2007"

    id = db.Column(db.Integer, primary_key=True)
    country = db.Column(db.String)
    pop = db.Column(db.Integer)
    continent = db.Column(db.String)
    lifeExp = db.Column(db.Float)
    gdpPercap = db.Column(db.Float)