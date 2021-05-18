# Course: EE551 Python for Engineer
# Author: Sushrut Madhavi
# Date: 2021/05/04
# Version: 1.0
# Defines routes for the front-end
from flask import render_template, url_for, flash, redirect, request, send_from_directory, send_file
from image_processor.forms import ImageProcessForm
from image_processor import app
from image_processor.algorithms.edge_detection import gaussian, sobel_edge, nonMax
from image_processor.algorithms.line_detection import hessian, RANSAC
import os

# Performs the specified operation on an image
# Checks if required output already exists in cache
# Inputs:
#   form_picture: file path of the image to be processed
#   form_process: the operation to be performed on the image
# Outputs:
#   processed_path: file path of the resulting image
#   filename: file name of the resulting image
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

# Route for home page which has the main functionality of the website
# Contains a form for image upload and space for displaying result
# Also displays an optional download button after processing
# Methods:
#   GET: Called when user navigates to this route. Image displays are blank in this case.
#   POST: Called when user submits the form. Image display areas show original image and the result with selected filter applied
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

# Downloads an image from storage to user's device 
# Inputs:
#   filename: file path of the image to be downloaded
@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.root_path, 'static/images', filename)
    return send_file(file_path, as_attachment=True)

# Prevents image from being fetched by the browser from its cache
# This is necessary to avoid the issue of wrong image being displayed in result area if the page is not refreshed
# It is accomplished by modifying some header values in the response object
# Inputs:
#   response: the response object from the server
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response