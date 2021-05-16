from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired, InputRequired, ValidationError
import os
import os.path
from image_processor import app

class ImageProcessForm(FlaskForm):
    image = FileField(label='Upload Image', validators=[FileAllowed(['jpg', 'png'])])
    algorithm = RadioField('Algorithm', choices=[('gaussian', 'Gaussian Filter'), ('sobel', 'Sobel Filter'), ('nonMax', 'Non-maximum Suppression'), ('hessian', 'Hessian Detection'), ('ransac', 'RANSAC')], validators=[DataRequired()])
    submit = SubmitField(label='Perform Process')

    def validate_image(self, image):
        if image.data is None:
            image_path = os.path.join(app.root_path, 'static/images/upload.png')
            if not os.path.isfile(image_path):
                raise ValidationError('Please upload an image!')
            