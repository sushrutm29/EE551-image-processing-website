from flask import Flask
from flask_mongoengine import MongoEngine
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = '2ee9596db295364f62dd6aef17a2e3ca'
app.config['MONGODB_SETTINGS'] = {
    'db': 'chore-russian-roulette-db',
    'host': 'localhost',
    'port': 27017
}
db = MongoEngine()
db.init_app(app)

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

from main import routes