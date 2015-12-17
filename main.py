import cv2
import numpy as np
import matplotlib.pyplot as plt


def prepare_image():
    image = cv2.imread('cat.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small_image = cv2.resize(
        gray_image, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    height, width = small_image.shape
    circle = np.zeros((height, width))
    cv2.circle(circle, (width/2, height/2), height/2, 1, thickness=-1)
    masked_data = small_image * circle
    extra_pixels = width - height
    cropped = masked_data[0:height, extra_pixels / 2:width - extra_pixels/2]
    cv2.imwrite('prepared.png', cropped)


def main():
    prepare_image()
    image = cv2.imread('dot.png', 0)
    rows, cols = image.shape
    # cv2.imshow('prepared', image)

    all_projected = {}

    angles = range(0, 360, 1)
    for i in angles:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), i, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        projected = rotated.sum(axis=0)
        all_projected[i] = projected
        # cv2.imshow('rotated {}'.format(i), rotated)
        # plt.plot(projected)

    # Build sinogram!
    height = len(all_projected.keys())
    width = len(all_projected[0])
    sinogram = np.zeros((height, width))
    for index, angle in enumerate(angles):
        sinogram[index] = all_projected[angle]

    plt.imshow(sinogram, cmap='gray', interpolation='bicubic')
    plt.show()

if __name__ == '__main__':
    main()
