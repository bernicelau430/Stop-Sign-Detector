# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2 # import the cv library
import numpy as np # import numerical library

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur to reduce noise
    canny_image = cv2.Canny(blur, 100, 200) # detect edges using canny function
    return (canny_image)

def region_of_interest(image): # define region of interest
    polygons = np.array([[(0,250), (0,200), (150,100), (500,100), (650,200), (650,250)]])
    mask = np.zeros_like(image) # create mask with some image dimension
    cv2.fillPoly(mask, polygons, 255) # fill in mask with white pixels
    masked_image = cv2.bitwise_and(image, mask) # apply mask to image
    return masked_image

def display_lines(image, lines): # create a line image
    line_image = np.zeros_like(image)
    if lines is not None: # check if lines are empty
        for line in lines: #loop through each line
            x1, y1, x2, y2 = line.reshape(4) # extract out coordinate of each line
            cv2.line(line_image, (x1,y1), (x2, y2), (0,255,0), 5) # plot each line
    return line_image

image = cv2.imread('track_image.jpg') # import picture from directory
lane_image = np.copy(image) # create a copy of the row picture
lane_image = cv2.resize(lane_image, (650, 500)) # resize the picture
# gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY) # convert image to grayscale
# blur = cv2.GaussianBlur(gray, (5, 5), 0) # apply Gaussian blur to reduce noise
# canny_image = cv2.Canny(blur, 100, 200) # 1:2 ratio
canny_image = canny(lane_image) # edge detection using the canny function
cropped_image = region_of_interest(canny_image) # crop image using ROI
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 50, np.array([]), minLinelength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines) # find average line
line_image = display_lines(lane_image, averaged_lines) # display lane lines

cv2.imshow('figure1', image) # display picture in a figure
cv2.waitKey(0) # hold figure open
cv2.destroyAllWindows()

def make_coordinates(image, line_parameters): # define coordinate for plotting lanes
    slope, intercept = line_parameters # extract out slope and intercept values
    y1 = 300 # choose starting y1 location to draw lanes
    y2 = 120 # chooise ending y2 location
    x1 = int((y1 - intercept)/slope) # calculate corresponding x1 value
    x2 = int((y2 - intercept)/slope) # calculate corresponding x2 value
    return np.array([x1, y1, x2, y2]) # return coordinate for lane to pilot

def average_slope_intercept(image, lines): # calculate avg slope and intercept
    left_fit = [] # declare left lane vector
    right_fit = [] # declare right lane vector
    for line in line: # looping through all detected lines
        x1, y1, x2, y2 = line.reshape(4) # extract out line coordinates
        parameters = np.polyfit((x1,x2), (y1,y2), 1) # linear fit through 2 points
        slope = parameters[0] # extract slope - index 0
        intercept = parameters[1] # extract intercept - index 1
        if slope < 0: # if slope is negative, append parameters to left lane
            left_fit.append([slope, intercept]) # appending parameters
        else: # if slope is positive, append parameters to the right lane
            right_fit.append([slope, intercept]) # append parameters to right lane
        left_fit_average = np.average(left_fit, axis=0) # calculate avg slope and intercept
        right_fit_average = np.average(right_fit, axis=0) # calculate avg slope and intercept
        left_line = make_coordinates(image, left_fit_average) # calculate lane coordinates
        right_line = make_coordinates(image, right_fit_average) # calculate lane coordinates
        return np.array([left_line, right_line]) # return left and right lanes

    cap = cv2.VideoCapture("video.mp4")
    while(cap.isOpened()):
        frame = cap.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLinelength=40, maxLineGap=5)
        averaged_lines = average_slope_interept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('figure', combo_image)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
