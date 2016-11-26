# ASSIGNMENT 2
# Chi Yang
# 903137667

import cv2
import numpy as np
import scipy as sp

""" Assignment 2 - Basic Image Input / Output / Simple Functionality

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.
"""

def numberOfPixels(image):
    """ This function returns the number of pixels in a grayscale image.

    Note: A grayscale image has one dimension as covered in the lectures. You
    DO NOT have to account for a color image.

    You may use any / all functions to obtain the number of pixels in the image.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        int: The number of pixels in an image.
    """
    # WRITE YOUR CODE HERE.  
    return image.size

    # END OF FUNCTION.

def averagePixel(image):
    """ This function returns the average color value of a grayscale image.

    Assignment Instructions: In order to obtain the average pixel, add up all
    the pixels in the image, and divide by the total number of pixels. We advise
    that you use the function you wrote above to obtain the number of pixels!

    You may not use numpy.mean or numpy.average. All other functions are fair
    game.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        int: The average pixel in the image (Range of 0-255).
    """
    # WRITE YOUR CODE HERE.
    
    return int(cv2.sumElems(image)[0]/numberOfPixels(image))

    # END OF FUNCTION.

def convertToBlackAndWhite(image):
    """ This function converts a grayscale image to black and white.

    Assignment Instructions: Iterate through every pixel in the image. If the
    pixel is strictly greater than 128, set the pixel to 255. Otherwise, set the
    pixel to 0. You are essentially converting the input into a 1-bit image, as 
    we discussed in lecture, it is a 2-color image.

    You may NOT use any thresholding functions provided by OpenCV to do this.
    All other functions are fair game.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The black and white image.
    """
    # WRITE YOUR CODE HERE.
    
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if image[i,j]>128:
                image[i,j]=255
            else:
                image[i,j]=0
    """
    cv2.imshow('white&black',image)
    cv2.waitKey(0)
    cv2.imwrite('white&black.jpg',image)
    """
    return image

    # END OF FUNCTION.

def averageTwoImages(image1, image2):
    """ This function averages the pixels of the two input images.

    Assignment Instructions: Obtain the average image by adding up the two input
    images on a per pixel basis and dividing them by two.

    You may use any / all functions to obtain the average image output.

    Note: You may assume image1 and image2 are the SAME size.

    Args:
        image1 (numpy.ndarray): A grayscale image represented in a numpy array.
        image2 (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The average of image1 and image2.

    """


    # WRITE YOUR CODE HERE.

    for i in range(0,image1.shape[0]):
        for j in range(0,image1.shape[1]):
            image1[i,j]=int(.5*image1[i,j]+.5*image2[i,j])
    """
    cv2.imshow('avg',image1)
    cv2.waitKey(0)
    cv2.imwrite('avg.jpg',image1)
    """
    return image1


    # END OF FUNCTION.

def flipHorizontal(image):
    """ This function flips the input image across the horizontal axis.

    Assignment Instructions: Given an input image, flip the image on the
    horizontal axis. This can be interpreted as switching the first and last
    column of the image, the second and second to last column, and so on.

    You may use any / all functions to flip the image horizontally.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The horizontally flipped image.

    """
    # WRITE YOUR CODE HERE.
    """
    flipped = image.copy()
    no_rows = flipped.shape[0]
    no_cols = flipped.shape[1]
    for row in range(no_rows):
        half = no_cols/2
        for col in range(half):
            pixl = flipped[row, col].copy()
            flipped[row, col] = flipped[row, no_cols - col - 1].copy()
            flipped[row, no_cols - col - 1] = pixl
    
    """
    flipped = cv2.flip(image, 1)
    """
    cv2.imshow('fliphorizontal',flipped)
    cv2.waitKey(0)
    cv2.imwrite('fliphorizontal.jpg',flipped)
    """
    
    return flipped


    # END OF FUNCTION.
