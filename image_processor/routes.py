from flask import render_template, url_for, flash, redirect, request
from image_processor.forms import ImageProcessForm
from image_processor import app
from image_processor.algorithms.edge_detection import gaussian, sobel, nonMax
import os

def process_picture(form_picture, form_process):
    processed_path = ""

    if form_process == "gaussian":
        gaussian(form_picture, 2)
        processed_path = url_for('static', filename='images/' + 'gaussian_img.png')
    if form_process == "sobel":
        sobel(form_picture)
        processed_path = url_for('static', filename='images/' + 'sobel_img.png')
    if form_process == "nonMax":
        nonMax(form_picture)
        processed_path = url_for('static', filename='images/' + 'non_max_img.png')

    return processed_path

@app.route("/", methods=['GET', 'POST'])
def home():
    form = ImageProcessForm()
    picture_path = url_for('static', filename='images/' + 'default_original.jpeg')
    processed_path = url_for('static', filename='images/' + 'default_original.jpeg')

    if form.validate_on_submit():
        picture_path = os.path.join(app.root_path, 'static/images', form.image.data.filename)
        form.image.data.save(picture_path)
        processed_path = process_picture(picture_path, form.algorithm.data)
        picture_path = url_for('static', filename='images/'+form.image.data.filename)
        
        return render_template('home.html', title='Image Processor', original_image=picture_path, processed_image=processed_path, form=form)

    return render_template('home.html', title='Image Processor', original_image=picture_path, processed_image=processed_path, form=form)