# Course: EE551 Python for Engineer
# Author: Sushrut Madhavi
# Date: 2021/05/04
# Version: 1.0
# Defines form object for image upload and process selection
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import RadioField, SubmitField
from wtforms.validators import DataRequired, InputRequired, ValidationError
import os
import os.path
from image_processor import app
from PIL import Image

# Creates the form object and defines its fields
# Also performs necessary validation checks to ensure correct data is received by the server
# Fields:
#   image: image file to be processed
#   algorithm: the filter to be applied to the image
#   submit: submit button to send image and name of filter to server for processing
# Validation:
#   image: Checks to make sure image was uploaded. If not, checks to make sure previously uploaded image exists in storage.
#           If neither is true, raises a validation error
#   algorithm: Ensures that one algorithm is selected before form submission
class ImageProcessForm(FlaskForm):
    image = FileField(label='Upload Image', validators=[FileAllowed(['jpg', 'png'])])
    algorithm = RadioField('Algorithm', choices=[('gaussian', 'Gaussian Filter'), ('sobel', 'Sobel Filter'), ('nonMax', 'Non-maximum Suppression'), ('hessian', 'Hessian Detection'), ('ransac', 'RANSAC')], validators=[DataRequired()])
    submit = SubmitField(label='Perform Process')

    def validate_image(self, image):
        if image.data is None:
            image_path = os.path.join(app.root_path, 'static/images/upload.png')
            if not os.path.isfile(image_path):
                raise ValidationError('Please upload an image!')
            