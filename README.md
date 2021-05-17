# Image Processing Website

This website allows image processing operations to be performed on an image of the users' choice. The supported operations are gaussian filter, sobel filter, non-maximum supression, hessian detection and RANSAC. It provides an easy-to-use interface for the same. More operations are expected to be added soon!

# Core Image Processing Filters
* [Gaussian Filter](http://www.justin-liang.com/tutorials/canny/#gaussian): smooths the uploaded image and removes any noise.
* [Sobel Filter](http://www.justin-liang.com/tutorials/canny/#gradient): processes and displays the gradient magnitude of the uploaded image.
* [Non-Max suppression](http://www.justin-liang.com/tutorials/canny/#suppression)
* [Hessian Filter](https://en.wikipedia.org/wiki/Hessian_matrix): calculates and thresholds each pixel’s determinant then displays the remaining non-zero pixels as the points of interest.
* [RANSAC Filter](https://en.wikipedia.org/wiki/Random_sample_consensus): randomly forms/finds four lines with the most inlier points from the hessian processed image.

# Getting Started
Please follow the below instructions to run the application locally on your system

# Prerequisites

* Visual Studio Code (Optional) - https://code.visualstudio.com/download
* Python 3 - https://www.python.org/download/releases/3.0/
* Browser - Google Chrome preferred

# Install and Run the project

1. Clone the repository and navigate to the project directory in a terminal window

2. Run the following command in a terminal window to install the required Python dependencies:

    ``` python3 -m pip install -r requirements.txt ```

3. Start the Flask server to run the application on local host using the following command:

    ``` python3 run.py ```
    
4. Access our website by pasting the following URL in a browser window (Ensure JavaScript is enabled):

    ```http://localhost:5000```

# Core Python Modules
* Flask - Micro web framework
* Numpy - Library for multi-dimensional arrays' operations
* Matplotlib - Library for plotting operations

# Authors
* Sushrut Madhavi
* Lun-Wei Chang (David)
