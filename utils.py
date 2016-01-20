# Description: A library of common utilities
# Author: Matt Ferraro
import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def imagify(fft):
    '''
    2D ffts usually have real and imaginary components, and they also usually
    have way too much dynamic range to be sensibly viewed. This function
    takes a 2D FFT and returns the log10 of the absolute value of the 2D FFT,
    so it is suitable for displaying as an image
    '''
    return np.log10(np.absolute(fft) + np.ones(fft.shape))


def pupil_function(width=11):
    '''
    Create a pupil: a square matrix with a raised cylinder of the specified
    width. The square is width * 2 + 1 pixels wide
    '''
    background = np.zeros((width * 2 + 1, width * 2 + 1))
    cv2.circle(background, (width, width), width / 2, 1, thickness=-1)
    return background


def pupil_to_transfer_function_fft(pupil_func):
    '''
    Given a pupil function, return the FFT of its transfer function

    This is achieved by just convolving the pupil function with itself
    '''
    return convolve2d(pupil_func, pupil_func, boundary='symm', mode='same')


def plot_image_and_slice(image, title=None):
    '''
    Given an image, plot the image itself and on a shared x axes, a single
    slight through the middle of the image
    '''
    plt.figure(1)
    image_plot = plt.subplot(210)
    plt.imshow(image, cmap='gray', interpolation='nearest')

    plt.subplot(211, sharex=image_plot)
    plt.plot(image[image.shape[0] / 2], 'ro', )
    if title:
        plt.title(title)
    plt.show()


def pad_to_match(reference_array, candidate_array):
    '''
    Given a reference array and a candidate array, pad the candidate array with
    zeros equally on all sides so that it is equal in size to the reference
    '''
    # This code is fragile and makes a lot of assumptions:
    #   - the reference and the candidate are both square
    #   - the reference image is larger than the candidate
    #   - the difference is a multiple of 2
    # The code should be improved to not rely on those assumptions
    padding = (reference_array.shape[0] - candidate_array.shape[0]) / 2
    return np.pad(candidate_array, padding, mode='constant')
