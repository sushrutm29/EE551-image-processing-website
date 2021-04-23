import cv2 as cv
import numpy as np
import math
import edge_detection as edge

# Sobel Filter method
def sobel(gaussian_img, type):
    # initializes the gradient X and Y matrices
    gradient_x = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    gradient_y = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    threshold = 10 #max value of pixel gradient magnitude
    img_rows = gaussian_img.shape[0]
    img_cols = gaussian_img.shape[1]
    gradient_magnitude = np.zeros([img_rows, img_cols], dtype = float)
    derivative = np.zeros([img_rows, img_cols], dtype = float)
    # calculates gradient magnitude for each pixel
    for i in range(img_rows - 2):
        for j in range(img_cols - 2):
            # gets the current 3x3 matrix from the original image
            tempImg = gaussian_img[i : i + 3, j : j + 3]
            # calculates the x-axis and y-axis gradient magnitude
            tempImg = tempImg.astype(float)
            Gx = np.sum(np.multiply(gradient_x, tempImg))
            Gy = np.sum(np.multiply(gradient_y, tempImg))
            # calculates derivative x and y
            if type == "derivative_x":
                derivative[i + 1][j + 1] = np.sum(Gx)
            else:
                derivative[i + 1][j + 1] = np.sum(Gy)
            # applies the gradient magnitude formula to current pixel
            gradient_magnitude[i + 1][j + 1] = math.sqrt(Gx**2 + Gy**2)

    # applies the threshold to the gradient magnitude matrix
    # sobelImg = max(gradient_magnitude, threshold)
    gradient_magnitude[gradient_magnitude < threshold] = threshold
    gradient_magnitude[gradient_magnitude == round(threshold)] = 0
    
    return gradient_magnitude, derivative

def hessian(img, threshold = 10):
    # throws an error if image is not provided
    if not img:
        raise ValueError("No image was provided!")
    # reads in the image
    current_img = edge.readImg(img)
    # applies Gaussian filter
    sigma = 2
    gaussian_img = edge.gaussian(img, sigma)
    # calculates Ixx, Ixy, Iyy matrix
    _, derivative_x = sobel(gaussian_img, 'derivative_x')
    _, derivative_y = sobel(gaussian_img, 'derivative_y')
    _, derivative_xx = sobel(derivative_x, 'derivative_x')
    _, derivative_yy = sobel(derivative_y, 'derivative_y')
    _, derivative_xy = sobel(derivative_x, 'derivative_y')

    # applies Hessian determinant formula
    determinant = np.subtract(np.multiply(derivative_xx, derivative_yy), np.multiply(derivative_xy, derivative_xy))
    edge.saveImg(determinant, "determinant_img01")
    # applies the threshold onto the determinant of the Hessian
    determinant[determinant < threshold] = threshold
    determinant[determinant == round(threshold)] = 0
    edge.saveImg(determinant, "determinant_img02")

    # applies the non-maximum suppression
    hessianImg = nonMaxSuppression(determinant)
    edge.saveImg(hessianImg, "hessian_img")

def nonMaxSuppression(img_mag):
    # initializes row and column variables
    rows = img_mag.shape[0]
    cols = img_mag.shape[1]
    # initalizes output image
    outputImg = np.zeros([rows, cols], dtype = float)
    # assigns edge orientation
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # compares the current pixel magnitude 
            if (img_mag[i + 1][j - 1] <= img_mag[i][j]) and \
                (img_mag[i + 1][j] <= img_mag[i][j]) and \
                (img_mag[i + 1][j + 1] <= img_mag[i][j]) and \
                (img_mag[i][j - 1] <= img_mag[i][j]) and \
                (img_mag[i][j + 1] <= img_mag[i][j]) and \
                (img_mag[i - 1][j - 1] <= img_mag[i][j]) and \
                (img_mag[i - 1][j] <= img_mag[i][j]) and \
                (img_mag[i - 1][j + 1] <= img_mag[i][j]):
                if img_mag[i][j] > 0:
                    outputImg[i][j] = 1
                else:
                    outputImg[i][j] = 0
                # outputImg[i][j] = 1 if img_mag[i][j] > 0 else 0
            else:
                outputImg[i][j] = 0
    return outputImg

def RANSAC(img, sigma):
    print("ransca")

def hough(img, type):
    print("hough")

def main():
    hessian('images/woman_original.jpg')

if __name__ == "__main__":
    main()