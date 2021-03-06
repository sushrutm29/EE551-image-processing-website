# Course: EE551 Python for Engineer
# Author: Sushrut Madhavi
# Date: 2021/04/25
# Version: 1.0
# Creates the flask app
from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '2ee9596db295364f62dd6aef17a2e3ca'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

from image_processor import routes