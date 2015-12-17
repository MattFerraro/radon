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
    image = cv2.imread('prepared.png', 0)
    rows, cols = image.shape
    # cv2.imshow('prepared', image)

    all_projected = {}

    for i in xrange(0, 360, 5):
        M = cv2.getRotationMatrix2D((cols/2, rows/2), i, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        projected = rotated.sum(axis=0)
        all_projected[i] = projected
        # cv2.imshow('rotated {}'.format(i), rotated)
        # plt.plot(projected)

    print all_projected[5]
    # plt.show()

    print "done"
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
