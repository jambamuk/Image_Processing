from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

import random as rng

kernel = np.ones((5, 5),np.uint8)
rng.seed(12345)


def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    print(len(contours))
    for i in range(5):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, sorted_contours, 1 , color, 2, -1, hierarchy, 0)
        #print(cv.contourArea(contours[2]))
    # Show in a window
    cv.imshow('Contours', drawing)



# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='HappyFish.jpg')
args = parser.parse_args()


src = cv.imread('Cow data/hsv_ranged_images/Cow_1.jpg', 0)
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.imread('Cow data/hsv_ranged_images/Cow_1.jpg', 0)
src_gray = cv.medianBlur(src_gray, 5)
src_gray = cv.morphologyEx(src_gray, cv.MORPH_OPEN, kernel)
src_gray = cv.morphologyEx(src_gray, cv.MORPH_CLOSE, kernel)
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()