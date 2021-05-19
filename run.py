# Course: EE551 Python for Engineer
# Author: Sushrut Madhavi
# Date: 2021/04/12
# Version: 1.0
# Main file to run the flask application
# This is the file to be run for starting the web server
from image_processor import app

if __name__ == '__main__':
    app.run(debug=True)