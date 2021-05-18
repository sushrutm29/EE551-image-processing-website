from flask import render_template, url_for, flash, redirect, request, send_from_directory, send_file
from image_processor.forms import ImageProcessForm
from image_processor import app
from image_processor.algorithms.edge_detection import gaussian, sobel_edge, nonMax
from image_processor.algorithms.line_detection import hessian, RANSAC
import os

def process_picture(form_picture, form_process):
    processed_path = ""
    filename = ""

    if form_process == "gaussian":
        processed_path = os.path.join(app.root_path, 'static/images/gaussian_img.png')
        filename = "gaussian_img.png"
        if not os.path.isfile(processed_path):
            gaussian(form_picture, 2)
        processed_path = url_for('static', filename='images/' + 'gaussian_img.png')
    if form_process == "sobel":
        processed_path = os.path.join(app.root_path, 'static/images/sobel_img.png')
        filename = "sobel_img.png"
        if not os.path.isfile(processed_path):
            sobel_edge(form_picture)
        processed_path = url_for('static', filename='images/' + 'sobel_img.png')
    if form_process == "nonMax":
        processed_path = os.path.join(app.root_path, 'static/images/non_max_img.png')
        filename = "non_max_img.png"
        if not os.path.isfile(processed_path):
            nonMax(form_picture)
        processed_path = url_for('static', filename='images/' + 'non_max_img.png')
    if form_process == 'hessian':
        processed_path = os.path.join(app.root_path, 'static/images/hessian_img.png')
        filename = "hessian_img.png"
        if not os.path.isfile(processed_path):
            hessian(form_picture)
        processed_path = url_for('static', filename='images/' + 'hessian_img.png')
    if form_process == 'ransac':
        processed_path = os.path.join(app.root_path, 'static/images/ransac_img.png')
        filename = "ransac_img.png"
        if not os.path.isfile(processed_path):
            hessian_img = hessian(form_picture)
            RANSAC(form_picture, hessian_img)
        processed_path = url_for('static', filename='images/' + 'ransac_img.png')

    return processed_path, filename

@app.route("/", methods=['GET', 'POST'])
def home():
    download = False
    form = ImageProcessForm()
    picture_path = url_for('static', filename='images/' + 'default_original.jpeg')
    processed_path = url_for('static', filename='images/' + 'default_original.jpeg')
    images_folder = os.path.join(app.root_path, 'static/images')

    if request.method == 'GET':
        for image_file in os.listdir(images_folder):
            if image_file != "default_original.jpeg":
                os.remove(os.path.join(images_folder, image_file))

    if form.validate_on_submit():
        download=True
        picture_path = os.path.join(app.root_path, 'static/images/upload.png')
        if form.image.data is not None:
            form.image.data.save(picture_path)
            for image_file in os.listdir(images_folder):
                if image_file != "default_original.jpeg" and image_file != "upload.png":
                    os.remove(os.path.join(images_folder, image_file))

        processed_path, filename = process_picture(picture_path, form.algorithm.data)
        picture_path = url_for('static', filename='images/upload.png')
        
        return render_template('home.html', title='Image Processor', original_image=picture_path, processed_image=processed_path, form=form, download=download, filename=filename)

    return render_template('home.html', title='Image Processor', original_image=picture_path, processed_image=processed_path, form=form, download=download)

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.root_path, 'static/images', filename)
    return send_file(file_path, as_attachment=True)

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response