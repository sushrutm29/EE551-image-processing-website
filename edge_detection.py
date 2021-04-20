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
            kernel[i, j] = math.exp(-(center_dist) / (2 * (sigma**2)))
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
    # gaussianImg = normalize(gaussianImg)
    #displays the image
    saveImg(gaussianImg, "gaussian_img")
    return gaussianImg

# Sobel Filter method
def sobel(gaussian_img):
    # initializes the gradient X and Y matrices
    gradient_x = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    gradient_y = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    threshold = 10 #max value of pixel gradient magnitude
    img_rows = gaussian_img.shape[0]
    img_cols = gaussian_img.shape[1]
    gradient_magnitude = np.zeros([img_rows, img_cols], dtype = float)
    gradient_direction = np.zeros([img_rows, img_cols], dtype = float)
    # calculates gradient magnitude for each pixel
    for i in range(img_rows - 2):
        for j in range(img_cols - 2):
            # gets the current 3x3 matrix from the original image
            tempImg = gaussian_img[i : i + 3, j : j + 3]
            # calculates the x-axis and y-axis gradient magnitude
            tempImg = tempImg.astype(float)
            Gx = np.sum(np.multiply(gradient_x, tempImg))
            Gy = np.sum(np.multiply(gradient_y, tempImg))
            # applies the gradient magnitude formula to current pixel
            gradient_magnitude[i + 1][j + 1] = math.sqrt(Gx**2 + Gy**2)
            # calculates the gradient direction
            gradient_direction[i + 1][j + 1] = math.degrees(math.atan(Gy / Gx)) 

    # applies the threshold to the gradient magnitude matrix
    # sobelImg = max(gradient_magnitude, threshold)
    gradient_magnitude[gradient_magnitude < threshold] = threshold
    sobelImg = gradient_magnitude
    sobelImg[sobelImg == round(threshold)] = 0
    # displays the sobel image
    saveImg(sobelImg, "sobel_img")
    return sobelImg, gradient_direction

# Non-maximum Suppression method
def nonMax(img_mag, g_direct):
    # initializes necessary variables
    rows = img_mag.shape[0]
    cols = img_mag.shape[1]
    # initializes degree variables used for the four directions
    h_right, h_left, v_right, v_left, half_circle = 22.5, 157.5, 67.5, 112.5, 180
    outputImg = np.zeros([rows, cols], dtype = float)
    pad_size = 1
    img_mag = np.lib.pad(img_mag, pad_size, 'symmetric')
    # assigns the edge orientation
    for i in range(1, rows - 2 + pad_size):
        for j in range(1, cols - 2 + pad_size):
            # horizontal direction
            if (g_direct[i][j] <= half_circle and g_direct[i][j] >= h_left) or \
                (g_direct[i][j] < -h_left and g_direct[i][j] >= -half_circle) or \
                    (g_direct[i][j] >= -h_right and g_direct[i][j] <= h_right):
                if (img_mag[i + 1][j] <= img_mag[i][j]) and \
                    (img_mag[i - 1][j] <= img_mag[i][j]):
                    outputImg[i][j] = img_mag[i][j]
                else:
                    outputImg[i][j] = 0
            # vertical direction
            elif (g_direct[i][j] < -v_right and g_direct[i][j] >= -v_left) or \
                (g_direct[i][j] <= v_left and g_direct[i][j] >= v_right):
                if (img_mag[i][j + 1] <= img_mag[i][j]) and \
                    (img_mag[i][j - 1] <= img_mag[i][j]):
                    outputImg[i][j] = img_mag[i][j]
                else:
                    outputImg[i][j] = 0
            # diagonal direction (-22.5 ~ -67.5 and 112.5 ~ 157.5)
            elif (g_direct[i][j] < -h_right and g_direct[i][j] >= -v_right) or \
                (g_direct[i][j] <= h_left and g_direct[i][j] >= v_left):
                if (img_mag[i + 1][j - 1] <= img_mag[i][j]) and \
                    (img_mag[i - 1][j + 1] <= img_mag[i][j]):
                    outputImg[i][j] = img_mag[i][j]
                else:
                    outputImg[i][j] = 0
            # diagonal direction (22.5 ~ 67.5 and -112.5 ~ -157.5)
            elif (g_direct[i][j] < -v_left and g_direct[i][j] >= -h_left) or \
                (g_direct[i][j] <= v_right and g_direct[i][j] >= h_right):
                if (img_mag[i + 1][j + 1] <= img_mag[i][j]) and \
                    (img_mag[i - 1][j - 1] <= img_mag[i][j]):
                    outputImg[i][j] = img_mag[i][j]
                else:
                    outputImg[i][j] = 0
    # displays the image
    saveImg(outputImg, "non_max_img")
    
# Save the output images to database
def saveImg(img, img_title):
    cv.imwrite(img_title + '.png', img)

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

def main():
    gaussianImg = gaussian('images/woman_original.jpg', 2)
    sobelImg, gradient_direction = sobel(gaussianImg)
    nonMax(sobelImg, gradient_direction)

if __name__ == "__main__":
    main()