import cv2 as cv
import numpy as np
import math

#reads in the image
def readImg(img_file_name):
    current_img = cv.imread(img_file_name)
    # height, width, channels = img.shape
    #dimensions = img.shape
    #channels = img.shape[2]
    # converts the image to gray scale if it is color
    if current_img.shape[2] > 1:
        current_img = cv.cvtColor(current_img, cv.COLOR_BGR2GRAY)
    # cv.imshow('read in image', current_img)
    # cv.waitKey(0)
    return current_img
    
# Gaussian Filter method
def gaussian(img, sigma):
    current_img = readImg(img)
    rows = current_img.shape[0]
    columns = current_img.shape[1]

    # initializes kernel size
    kernel_size = 6 * sigma + 1
    kernel = np.zeros([kernel_size, kernel_size], dtype = float)
    # calculates center element's coordinates
    middleIndex = math.ceil(kernel_size / 2)
    # calculates kernel matrix
    for i in range(kernel_size):
        for j in range(kernel_size):
            center_dist = pow((i + 1 - middleIndex), 2) + \
                pow((j + 1 - middleIndex), 2)
            k_row = i + 1
            k_col = j + 1
            kernel[i, j] = math.exp(-(center_dist) / (2 * sigma^2))
    kernel = (1 / (2 * math.pi * (sigma ** 2))) * kernel
    # applies wrap around padding to the original image
    pad_size = math.floor(kernel_size / 2)
    paddedImg = np.lib.pad(current_img, pad_size, 'symmetric')
    gaussianImg = np.zeros([rows, columns], dtype = float)
    k_rows = kernel.shape[0]
    k_cols = kernel.shape[1]
    # applies the Gaussian filter
    for i in range(rows):
        for j in range(columns):
            temp_matrix = paddedImg[i : i + k_rows, j : j + k_cols]
            temp_matrix = temp_matrix.astype(float)
            temp_matrix = kernel * temp_matrix
            gaussianImg[i, j] = np.sum(temp_matrix)
    gaussianImg = normalize(gaussianImg)
    #displays the image
    displayImg(gaussianImg, "gaussian image")
# Sobel Filter method
def sobel():
    print("Sobel")

# Non-maximum Suppression method
def nonMax():
    print("non-max")

# Displays the output images and corresponding messages
def displayImg(img, img_title):
    # Reverses normalization before saving the image
    img_rev_norm = cv.convertScaleAbs(img, alpha=(255.0))
    cv.imwrite('test_img.png', img_rev_norm)

    # Displays the normalized image
    cv.imshow(img_title, img)
    cv.waitKey(0)
    
# Save the output images to database
def saveImg():
    print("")

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

def main():
    gaussian('images/woman_original.jpg', 3)

    # #waits for user to press "Esc" to close the displayed image window
    # cv.waitKey(0)

if __name__ == "__main__":
    main()