import cv2
import matplotlib.pyplot as plt
import radon
import argparse
import numpy as np


def main(img_name, num_slices):
    radon.prepare_image('cat.jpg', 'prepared.png')
    image = cv2.imread(img_name, 0)
    sinogram = radon.radon_transform(image, num_slices)
    high_passed = radon.high_pass(sinogram)
    # badly_reconstructed = radon.back_project(sinogram)
    reconstructed = radon.back_project(high_passed)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image, cmap='gray', interpolation='bicubic')

    # plt.figure(2)
    plt.subplot(122)
    plt.imshow(reconstructed, cmap='gray', interpolation='bicubic')
    plt.show()

    print "min max", np.amin(image), np.amax(image)
    print "min max", np.amin(reconstructed), np.amax(reconstructed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', default='prepared.png')
    parser.add_argument('--num-slices', '-n', default=36, type=int)
    args = parser.parse_args()
    main(args.image, args.num_slices)
