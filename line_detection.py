import cv2 as cv
import numpy as np
import math
import edge_detection as edge
import matplotlib.pyplot as plt
import random

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

def hessian(img, threshold = 1200):
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
    # applies the threshold onto the determinant of the Hessian
    determinant[determinant < threshold] = threshold
    determinant[determinant == round(threshold)] = 0

    # applies the non-maximum suppression
    hessianImg = nonMaxSuppression(determinant)
    edge.saveImg(hessianImg, "hessian_img")
    return hessianImg

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
                outputImg[i][j] = img_mag[i][j]
                outputImg[i][j] = 255 if img_mag[i][j] > 0 else 0
            else:
                outputImg[i][j] = 0
    return outputImg

def RANSAC(original_img, hessian_img, threshold = 5, minInlier = 60):
    # initializes RANSAC parameters
    numOfRandomPoints = 2 # number of random points selected for each model
    totalNumOfLines = 4 # total number of output lines
    numOfNonZeroPoints = np.count_nonzero(hessian_img) # total number of non-zero value pixels
    maxSampleNum = float('inf') # umber of samples N (number of iterations)
    iterationCount = 0 # number of iteration
    outputLineCount = 1
    levelOfConf = 0.5 # level of confidence to our model
    bestInlierRatio = 0.065
    # finds the non-zero value pixels' coordinates
    rows, cols = np.nonzero(hessian_img)[0], np.nonzero(hessian_img)[1]
    non_zero_points = np.vstack([rows, cols]).astype(float).T
    # initializes matrices used to store visited pixels
    selected_points = np.zeros([numOfNonZeroPoints, 2], dtype = float)
    selected_points_row = np.zeros(numOfNonZeroPoints, dtype = int)

    # initializes dictionary to store start, end and inlier lines
    outputLines = {}     
    # scans through the non-zero value pixels to find lines
    while maxSampleNum >= iterationCount:
        # selects two random points for the model
        max_val = numOfNonZeroPoints
        point_one = randomPoint(max_val, non_zero_points)
        point_two = randomPoint(max_val, non_zero_points)
        if point_two[1] == point_one[1]:
            continue
        # calculates the slope and y_intercept of the model line
        slope = (point_two[0] - point_one[0]) \
            / (point_two[1] - point_one[1])
        y_intercept = point_two[0] - (point_two[1] * slope)
        inliner_count = 0 # number of inliers that fit the current model
        i = 0
        # scans through all the non-zero value pixels to find inlier
        while i < len(rows):
            if math.isnan(non_zero_points[i][1]) and \
                math.isnan(non_zero_points[i][0]):
                i += 1
                continue
            # gets the current pixel's coordinates
            x1 = non_zero_points[i][1]
            y1 = non_zero_points[i][0]
            # finds the intercept coordinates
            x2 = (x1 + slope * y1 - slope * y_intercept) / (1 + slope ** 2)
            y2 = (((slope * x1) + ((slope ** 2) * y1) - \
                ((slope ** 2) * y_intercept)) / (1 + slope ** 2)) + y_intercept
            # calculates the distance from current point to the model line
            distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
            # determines whether distance is inlier or not
            if threshold >= distance:
                selected_points[i][0] = y1
                selected_points[i][1] = x1
                selected_points_row[i] = i
                inliner_count += 1
            i += 1
        
        # calculates inlier ratio
        inlierRatio = inliner_count / numOfNonZeroPoints
        # found the highest inlier ratio so far
        if inlierRatio == 0:
            # increments the iteration counter
            iterationCount += 1
            continue
        # print("inlierRatio = " + str(inlierRatio))
        if inlierRatio >= bestInlierRatio:
            # bestInlierRatio = inlierRatio
            # calculates proporation of outliers e
            outlierRatio = 1 - inlierRatio
            maxSampleNum = math.log(1 - levelOfConf) \
                / math.log(1 - ((1 - outlierRatio) ** numOfRandomPoints))
            outputLines[outputLineCount] = {'inlierRatio' : inlierRatio, \
                'startP' : point_one, 'endP' : point_two, 'slope' : slope, \
                'y_intercept' : y_intercept, 'inliers' : selected_points}
            outputLineCount += 1
            # sets the selected points to NaN in non-zero-points matrix
            nonZeroSelectedPointsRow = selected_points_row.nonzero()[0]
            for j in range(len(nonZeroSelectedPointsRow)):
                currentRow = nonZeroSelectedPointsRow[j]
                non_zero_points[currentRow, :] = float('nan')
        # increments the iteration counter
        iterationCount += 1
        # resets selected points matrix
        selected_points = np.zeros([numOfNonZeroPoints, 2], dtype = float)
        selected_points_row = np.zeros([numOfNonZeroPoints, 1], dtype = int)     
    
    # sorts the outputLines dictionary by inlier ratio
    sortedLines = sorted(outputLines.items(), key = lambda x : x[1]['inlierRatio'], reverse = True)
    # deletes extra lines stored in outputLines dictionary
    tempDict = {} # uses to store the 
    for i in range(min(len(sortedLines), totalNumOfLines)):
        tempDict[i] = {'inlierRatio' : sortedLines[i][1]['inlierRatio'], \
            'startP' : sortedLines[i][1]['startP'], \
            'endP' : sortedLines[i][1]['endP'], 'slope' : sortedLines[i][1]['slope'], \
            'y_intercept' : sortedLines[i][1]['y_intercept'], 'inliers' : sortedLines[i][1]['inliers']}
    outputLines = tempDict
    # plots the RANSAC image
    plotRANSAC(original_img, non_zero_points, outputLines, 0, hessian_img.shape[1], 0, hessian_img.shape[0])

def plotRANSAC(originalImg, non_zero_points, line_dict, x_min, x_max, y_min, y_max):
    normal_point_size = 9
    large_point_size = 15
    start_end_size = 25
    normal_color = (0,0,0)
    # draws the original image as background
    original_img = plt.imread(originalImg)
    fig, ax = plt.subplots()
    currentFigure = plt.gca
    currentFigure.Visible = "On"
    axes = plt.gca()
    plt.title("RANSAC")
    plt.axis([x_min, x_max, y_max, y_min])
    ax.imshow(original_img)
    plt.scatter(non_zero_points[:, 1], non_zero_points[:, 0], zorder = 1, marker = 'o', c = 'white', s = normal_point_size)
    
    # draws each line and its corresponding points
    default_colors = ["#40e1fd", "green", "#842bd7", "yellow", "#70fe42"]
    for i in range(len(line_dict)):
        # gets the x-axis limits
        m = line_dict[i]['slope']
        b = line_dict[i]['y_intercept']
        x_vals = np.array(axes.get_xlim())
        y_vals = b + m * x_vals
        current_color = default_colors[i]
        # removes the (0, 0) points from all rows and columns
        currentInliers = line_dict[i]['inliers']
        currentInliers = np.delete(currentInliers,np.where(~currentInliers.any(axis=1))[0], axis=0)
        # plots all the inlier points belong to the current line
        plt.scatter(currentInliers[:, 1], currentInliers[:, 0], zorder = 2, marker = 'd', c = current_color, s = normal_point_size)
        # makes the start and end points red
        startP = line_dict[i]['startP']
        endP = line_dict[i]['endP']
        plt.scatter(startP[1], startP[0], zorder = 4, marker = 's', c = 'red', s = large_point_size)
        plt.scatter(endP[1], endP[0], zorder = 4, marker = 's', c = 'red', s = large_point_size)
    
        plt.plot(x_vals, y_vals, zorder = 3, c = current_color)

    plt.show()

def randomPoint(max_val, input_matrix):
    # selects a random row number out of the non-zero-points set
    rand_row = np.random.randint(max_val)
    temp_x = input_matrix[rand_row, 0]
    temp_y = input_matrix[rand_row, 1]
    # returns a random point
    return [temp_x, temp_y, rand_row]

def main():
    uploadImg = 'images/road.png'
    hessian_img = hessian(uploadImg)
    RANSAC(uploadImg, hessian_img, threshold = 10, minInlier = 60)

if __name__ == "__main__":
    main()