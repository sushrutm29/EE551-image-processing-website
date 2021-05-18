# Overview
This website allows image processing operations to be performed on an image of the users' choice. The supported operations are gaussian filter, sobel filter, non-maximum supression, hessian detection and RANSAC. It provides an easy-to-use interface for the same. More operations are expected to be added soon!

# Supported Image Processing Filters
* [Gaussian Filter](http://www.justin-liang.com/tutorials/canny/#gaussian): smooths the uploaded image and reduces noise.
* [Sobel Filter](http://www.justin-liang.com/tutorials/canny/#gradient): processes and displays the edges of the uploaded image.
* [Non-Max suppression](http://www.justin-liang.com/tutorials/canny/#suppression): thins out the edge line generated by the sobel filter.
* [Hessian Filter](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/quadratic-approximations/a/the-hessian): calculates and thresholds each pixel’s determinant then displays the remaining non-zero pixels as the corners in the image.
* [RANSAC Filter](https://www.mathworks.com/discovery/ransac.html): randomly forms/finds four lines with the most inlier points from the hessian processed image.

# Getting Started
Please follow the below instructions to run the application locally on your system

# Prerequisites
* Visual Studio Code (Optional) - https://code.visualstudio.com/download
* Python 3 - https://www.python.org/download/releases/3.0/
* Browser - Google Chrome preferred

# Core Python Modules
* Flask - Micro web framework
* Numpy - Library for multi-dimensional arrays' operations
* Matplotlib - Library for plotting operations

# Install and Run the project

1. Clone the repository and navigate to the project directory in a terminal window

2. Run the following command in a terminal window to install the required Python dependencies:

    ``` python3 -m pip install -r requirements.txt ```

3. Start the Flask server to run the application on local host using the following command:

    ``` python3 run.py ```
    
4. Access our website by pasting the following URL in a browser window (Ensure JavaScript is enabled):

    ```http://localhost:5000```

5. Upload an image and select the filter to be applied

6. Press the submit button

7. Wait for the image to be processed and then download the output if needed

If you wish to apply another filter to the same image, just select the filter and press submit again.
No need to re-upload it!

# Authors
* Sushrut Madhavi
* Lun-Wei Chang (David)

# Example outputs
1. Original Image 
![original_img](https://github.com/sushrutm29/EE551-image-processing-website/blob/develop/sample_outputs/original_img.png)
2. Gaussian Output 
![gaussian_img](https://github.com/sushrutm29/EE551-image-processing-website/blob/develop/sample_outputs/gaussian_img.png)
3. Sobel Output 
![sobel_img](https://github.com/sushrutm29/EE551-image-processing-website/blob/develop/sample_outputs/sobel_img.png)
4. Hessian Output 
![hessian_img](https://github.com/sushrutm29/EE551-image-processing-website/blob/develop/sample_outputs/hessian_img.png)
5. RANSAC Output 
![ransac_img](https://github.com/sushrutm29/EE551-image-processing-website/blob/develop/sample_outputs/ransac_img.png)
