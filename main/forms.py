from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired

class ImageProcessForm(FlaskForm):
    def __init__(self):
        self.image = FileField('Upload Image', validators=[DataRequired(), FileAllowed(['jpg', 'png'])])
        self.process = RadioField('Select Process', validators=[DataRequired()] , choices=[('edge', 'Edge Detection'), ('line', 'Line Detection')], coerce=str)
        self.submit = SubmitField('Perform Process')