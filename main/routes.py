from flask import render_template, url_for, flash, redirect, request
from main.forms import ImageProcessForm
from main import app
import os

def process_picture(form_picture, form_process):
    picture_path = os.path.join(app.root_path, 'static/images', form_picture.filename)
    form_picture.save(picture_path)

    return picture_path

@app.route("/", methods=['GET', 'POST'])
def home():
    form = ImageProcessForm()
    picture_file = url_for('static', filename='images/' + 'default_original.jpeg')
    processed_file = url_for('static', filename='images/' + 'default_original.jpeg')

    if form.validate_on_submit():
        picture_file = process_picture(form.image.data, form.process.data)
        
        return render_template('home.html', title='Image Processor', original_image=picture_file, processed_image=processed_file, form=form)

    return render_template('home.html', title='Image Processor', original_image=picture_file, processed_image=processed_file, form=form)