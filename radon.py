import cv2
import numpy as np


def prepare_image(input_name, output_name):
    '''
    Given an input image, prepare it for use in radon related stuff.
    That means:
        -cast to grayscale
        -resize to be small
        -mask out all but a central circle
        -save as output file name
    '''
    image = cv2.imread(input_name)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(
        gray_image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    height, width = small_image.shape
    circle = np.zeros((height, width))
    cv2.circle(circle, (width/2, height/2), height/2, 1, thickness=-1)
    masked_data = small_image * circle
    extra_pixels = width - height
    cropped = masked_data[0:height, extra_pixels / 2:width - extra_pixels/2]
    cv2.imwrite(output_name, cropped)


def radon_transform(image):
    '''
    Perform the radon transform on an image, returning the sinogram
    '''
    rows, cols = image.shape
    angles = range(0, 180, 1)
    height = len(angles)
    width = cols
    sinogram = np.zeros((height, width))
    for index, alpha in enumerate(angles):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), alpha, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        sinogram[index] = rotated.sum(axis=0)
    return sinogram


def fft(signal):
    # Forward transform: f(x) -> F(k)
    fk = np.fft.rfft(signal)
    npts = len(signal)
    # Normalization -- the '2' here comes from the fact that we are
    # neglecting the negative portion of the frequency space, since
    # the FFT of a real function contains redundant information, so
    # we are only dealing with 1/2 of the frequency space.
    norm = 2.0/npts
    fk = fk*norm
    return fk


def ifft(signal):
    npts = (len(signal) - 1) * 2
    # Normalization -- the '2' here comes from the fact that we are
    # neglecting the negative portion of the frequency space, since
    # the FFT of a real function contains redundant information, so
    # we are only dealing with 1/2 of the frequency space.
    norm = 2.0/npts

    # Inverse transform: F(k) -> f(x) -- without the normalization
    fkinv = np.fft.irfft(signal/norm)
    return fkinv


def low_pass(sinogram):
    '''
    Given a sinogram, return a low pass filtered copy of it
    '''
    output = np.zeros(sinogram.shape)
    for index, row in enumerate(sinogram):
        fk = fft(row)
        ramp = np.arange(len(fk)) * 1.0  / len(fk)
        low_passed = fk * ramp
        output[index] = ifft(low_passed)
    return output


def back_project(sinogram):
    '''
    Given a sinogram, return the back-projection of it
    '''
    rotation_angle = 180 / len(sinogram)
    width = height = len(sinogram[0])
    reconstructed = np.zeros((width, height))
    for index, projection in enumerate(sinogram):
        M = cv2.getRotationMatrix2D(
            (width/2, height/2), -rotation_angle * index, 1)
        scaled_projection = projection / height
        back_projected = np.zeros((width, height))
        for row in back_projected:
            row += scaled_projection
        reconstructed += cv2.warpAffine(back_projected, M, (width, height))

    return reconstructed
