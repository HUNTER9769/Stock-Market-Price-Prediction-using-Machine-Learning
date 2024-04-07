from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_utils import database_exists


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///registerdata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

def database_uri():
        return app.config['SQLALCHEMY_DATABASE_URI']

if not database_exists(database_uri()):
            app.app_context().push()
            db.create_all()
            db.session.commit()


class Register(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return f"Register('{self.username}', '{self.email}', '{self.password}')"

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        register = Register(username=username, email=email, password=password)
        db.session.add(register)
        db.session.commit()
        return 'You are successfully registered'
    
@app.route('/users', methods=['GET', 'POST'])
def users():
      users = Register.query.all()
      return render_template('users.html', users=users)


if __name__ == "__main__":
    app.run(debug=True)