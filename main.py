import cv2
import matplotlib.pyplot as plt
import radon
import argparse
import pylab
import test


def main(img_name):
    radon.prepare_image('cat.jpg', 'prepared.png')
    image = cv2.imread(img_name, 0)
    sinogram = radon.radon_transform(image)
    low_passed = radon.low_pass(sinogram)
    badly_reconstructed = radon.back_project(sinogram)
    reconstructed = radon.back_project(low_passed)

    # pylab.clf()
    # pylab.subplot(221)
    # pylab.imshow(sinogram, cmap='gray', interpolation='bicubic')
    # pylab.subplot(222)
    # pylab.imshow(badly_reconstructed, cmap='gray', interpolation='bicubic')


    # pylab.subplot(223)
    # pylab.imshow(low_passed, cmap='gray', interpolation='bicubic')
    # pylab.subplot(224)
    # pylab.imshow(reconstructed, cmap='gray', interpolation='bicubic')
    # plt.show()

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(image, cmap='gray', interpolation='bicubic')

    # plt.figure(2)
    plt.subplot(122)
    plt.imshow(reconstructed, cmap='gray', interpolation='bicubic')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='prepared.png')
    args = parser.parse_args()
    main(args.image)
