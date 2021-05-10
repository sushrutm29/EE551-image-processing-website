from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '2ee9596db295364f62dd6aef17a2e3ca'

from image_processor import routes