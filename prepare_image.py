from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def main():
    print "hi"
    fourierSpectrumExample("dot.png")
    # pupil = create_pupil_function(width=15, cylinder_width=6)
    # cone = pupil_to_transfer_function_fft(pupil)
    # plot_pupil_and_transfer(pupil, cone)


def create_pupil_function(width=31, cylinder_width=11):
    '''
    Create a pupil function which is a square with the specified width,
    with an inner circle of the specified cylinder_width

    Hint: width should probably be odd
    '''
    circle = np.zeros((width, width))
    cv2.circle(circle, (width/2, width/2), cylinder_width/2, 1, thickness=-1)
    return circle


def pupil_to_transfer_function_fft(pupil_func):
    '''
    Given a pupil function, return the FFT of its transfer function

    This is achieved by just convolving the pupil function with itself
    '''
    return convolve2d(pupil_func, pupil_func, boundary='symm', mode='same')


def plot_pupil_and_transfer(pupil, transfer_fft):
    '''
    Given a pupil and a transfer function FFT, plot them
    '''
    plt.figure(1)
    pupil_plot = plt.subplot(222)
    plt.imshow(pupil, cmap='gray', interpolation='nearest')

    plt.subplot(220, sharex=pupil_plot)
    plt.plot(pupil[pupil.shape[0] / 2], 'ro', )

    cone_plot = plt.subplot(221)
    plt.imshow(transfer_fft, cmap='gray', interpolation='nearest')

    plt.subplot(223, sharex=cone_plot)
    plt.plot(transfer_fft[transfer_fft.shape[0] / 2], 'ro')
    plt.show()


def fourierSpectrumExample(filename):
    A = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    unshiftedfft = fft2(A)
    shiftedfft = fftshift(unshiftedfft)
    spectrum = np.log10(np.absolute(shiftedfft) + np.ones(A.shape))
    reconstructed = np.real(ifft2(unshiftedfft))

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(A, cmap='gray', interpolation='bicubic')

    plt.subplot(122)
    plt.imshow(spectrum, cmap='gray', interpolation='bicubic')
    plt.show()

    '''
    Start with a square image, apply a circular mask to it
        -this gives us something that is radially symmetric
        -only occupy the central 2/3 of the image
    Then simulate the optic:
        -Make a cone, multiply by FFT, then ifft, should see a blurred version of image
        -If source image is a single point, the IFFT should show an airy pattern
    '''


if __name__ == '__main__':
    main()
