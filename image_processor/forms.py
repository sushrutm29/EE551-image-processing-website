from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired, InputRequired

class ImageProcessForm(FlaskForm):
    image = FileField(label='Upload Image', validators=[DataRequired(), FileAllowed(['jpg', 'png'])])
    algorithm = RadioField('Algorithm', choices=[('gaussian', 'Gaussian Filter'), ('sobel', 'Sobel Filter'), ('nonMax', 'Non-maximum Suppression')], validators=[DataRequired()])
    submit = SubmitField(label='Perform Process')