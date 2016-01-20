# Description: demonstate making an airy disk, aka simulate a telescope
# Author: Matt Ferraro
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import cv2

import argparse
import utils


def main(image_name):
    # Make the pupil
    pupil = utils.pupil_function(35)
    utils.plot_image_and_slice(pupil, "Pupil")

    # Convolve it with itself to get our "cone"
    cone = utils.pupil_to_transfer_function_fft(pupil)
    utils.plot_image_and_slice(cone, "Cone")

    # Read in an image to work on
    original = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

    # Most of our work will be in frequency space
    shifted_fft = fftshift(fft2(original))

    # Our cone is the wrong size to use on our fft, so we need to pad it
    padded_cone = utils.pad_to_match(shifted_fft, cone)

    reconstructed = np.real(ifft2(ifftshift(padded_cone * shifted_fft)))
    utils.plot_image_and_slice(reconstructed, "Airy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='dot.png')
    args = parser.parse_args()
    main(args.image)
