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
    sigma = 3
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

def RANSAC(original_img, hessian_img, threshold = 10, minInlier = 60):
    # initializes RANSAC parameters
    numOfRandomPoints = 2 # number of random points selected for each model
    totalNumOfLines = 4 # total number of output lines
    numOfNonZeroPoints = np.count_nonzero(hessian_img) # total number of non-zero value pixels
    maxSampleNum = float('inf') # umber of samples N (number of iterations)
    iterationCount = 0 # number of iteration
    outputLineCount = 1
    levelOfConf = 0.99 # level of confidence to our model
    bestInlierRatio = 0
    # finds the non-zero value pixels' coordinates
    rows, cols = np.nonzero(hessian_img)[0], np.nonzero(hessian_img)[1]
    non_zero_points = np.vstack([rows, cols]).astype(float).T
    # initializes matrices used to store visited pixels
    selected_points = np.zeros([numOfNonZeroPoints, 2], dtype = float)
    selected_points_row = np.zeros([numOfNonZeroPoints, 1], dtype = int)
    
    # initializes dictionary to store start, end and inlier lines
    outputLines = {}     
    # scans through the non-zero value pixels to find lines
    while maxSampleNum >= iterationCount:
        # selects two random points for the model
        point_one = randomPoint(len(rows), non_zero_points)
        point_two = randomPoint(len(rows), non_zero_points)
        # skips the current point
        if point_two[1] == point_one[1]:
            iterationCount += 1
            continue
        # calculates the slope and y_intercept of the model line
        slope = (point_two[0].item() - point_one[0].item()) \
            / (point_two[1].item() - point_one[1].item())
        y_intercept = point_two[0].item() - (point_two[1].item() * slope)
        inliner_count = 0 # number of inliers that fit the current model
        i = 0
        # scans through all the non-zero value pixels to find inlier
        while i < len(rows):
            if non_zero_points[i][1] == float('nan') and \
                non_zero_points[i][0] == float('nan'):
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
            distance = math.sqrt((x2 - x1) ** 2  + (y2 - y1) ** 2)
            # determines whether distance is inlier or not
            if threshold >= distance:
                selected_points[i][0] = y1
                selected_points[i][1] = x1
                selected_points_row[i][0] = i
                inliner_count += 1
            i += 1

        # calculates inlier ratio
        inlierRatio = inliner_count / numOfNonZeroPoints
        # found the highest inlier ratio so far
        if inlierRatio == 0:
            # increments the iteration counter
            iterationCount += 1
            continue
        if inlierRatio >= bestInlierRatio:
            bestInlierRatio = inlierRatio
            # calculates proporation of outliers e
            outlierRatio = 1 - inlierRatio
            maxSampleNum = math.log(1 - levelOfConf) \
                / math.log(1 - (1 - outlierRatio) ** numOfRandomPoints)
            outputLines[outputLineCount] = {'inlierRatio' : inlierRatio, \
                'startP' : point_one, 'endP' : point_two, 'slope' : slope, \
                'y_intercept' : y_intercept, 'inliers' : selected_points}
            outputLineCount += 1
            # sets the selected points to NaN in non-zero-points matrix
            nonZeroSelectedPointsRow = selected_points_row.nonzero()
            for j in range(len(nonZeroSelectedPointsRow)):
                currentRow = nonZeroSelectedPointsRow[j]
                non_zero_points[currentRow, :] = float('nan')
            # increments the iteration counter
            iterationCount += 1
            # resets selected points matrix
            selected_points = np.zeros([numOfNonZeroPoints, 2], dtype = float)
            selected_points_row = np.zeros([numOfNonZeroPoints, 1], dtype = int)     
    # sorts outputLines dictionary
    # needs to manually sort the dictionary and remove the extra line
    # sorts the outputLines dictionary by inlier ratio
    sortedLines = sorted(outputLines.items(), key = lambda x : x[1]['inlierRatio'], reverse = True)
    # deletes extra lines stored in outputLines dictionary
    tempDict = {} # uses to store the 
    for i in range(len(sortedLines)):
        tempDict[i] = {'inlierRatio' : sortedLines[i][1]['inlierRatio'], \
            'startP' : sortedLines[i][1]['startP'], \
            'endP' : sortedLines[i][1]['endP'], 'slope' : sortedLines[i][1]['slope'], \
            'y_intercept' : sortedLines[i][1]['y_intercept'], 'inliers' : sortedLines[i][1]['inliers']}
    outputLines = tempDict
    # plots the RANSAC image
    plotRANSAC(original_img, non_zero_points, outputLines, 0, hessian_img.shape[1], 0, hessian_img.shape[0])

def plotRANSAC(originalImg, non_zero_points, line_dict, x_min, x_max, y_min, y_max):
    # 548 * 407
    normal_point_size = 9
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
    # plt.scatter(non_zero_points[:, 1], non_zero_points[:, 0], marker = 'o', c = 'yellow', s = normal_point_size)
    # draws each line and its corresponding points
    print("line_dict length = " + str(len(line_dict)))
    for i in range(len(line_dict)):
        # {'inlierRatio', 'startP', 'endP', 
        # 'slope', 'y_intercept', 'inliers'}
        # gets the x-axis limits
        m = line_dict[i]['slope']
        b = line_dict[i]['y_intercept']
        x_vals = np.array(axes.get_xlim())
        y_vals = b + m * x_vals
        current_color = (random.random(), random.random(), random.random())
        plt.plot(x_vals, y_vals, c = current_color)
        # removes the (0, 0) points from all rows and columns
        currentInliers = line_dict[i]['inliers']
        currentInliers = np.delete(currentInliers,np.where(~currentInliers.any(axis=1))[0], axis=0)
        # print(currentInliers)
        # plots all the inlier points belong to the current line
        plt.scatter(currentInliers[:, 1], currentInliers[:, 0], marker = 'd', c = current_color, s = normal_point_size)
        # makes the start and end points red
        startP = line_dict[i]['startP']
        endP = line_dict[i]['endP']
        plt.scatter(startP[1], startP[0], marker = 'd', c = 'red', s = normal_point_size)
        plt.scatter(endP[1], endP[0], marker = 'd', c = 'red', s = normal_point_size)
    
    plt.scatter(non_zero_points[:, 1], non_zero_points[:, 0], marker = 'o', c = 'yellow', s = normal_point_size)
    plt.show()

def testPlot():
    np.random.seed(19680801)
    N = 100
    r0 = 0.6
    x = 0.9 * np.random.rand(N)
    y = 0.9 * np.random.rand(N)
    area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
    c = np.sqrt(area)
    r = np.sqrt(x ** 2 + y ** 2)
    area1 = np.ma.masked_where(r < r0, area)
    area2 = np.ma.masked_where(r >= r0, area)
    plt.scatter(x, y, s=area1, marker='^', c=c)
    plt.scatter(x, y, s=area2, marker='o', c=c)
    # Show the boundary between the regions:
    theta = np.arange(0, np.pi / 2, 0.01)
    plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

    plt.show()

def randomPoint(max_value, input_matrix):
    # selects a random row number out of the non-zero-points matrix
    rand_row = np.random.randint(max_value)
    temp_x = input_matrix[rand_row, 0]
    temp_y = input_matrix[rand_row, 1]
    # returns a random point
    return [temp_x, temp_y]

def hough(img, type):
    print("hough")

def main():
    uploadImg = 'images/road.png'
    hessian_img = hessian(uploadImg)
    RANSAC(uploadImg, hessian_img, threshold = 10, minInlier = 60)
    # testPlot()

if __name__ == "__main__":
    main()