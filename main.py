import cv2
import matplotlib.pyplot as plt
import radon
import argparse


def main(img_name):
    # radon.prepare_image('cat.jpg', 'prepared.png')
    image = cv2.imread(img_name, 0)
    sinogram = radon.radon_transform(image)
    reconstructed = radon.back_project(sinogram)

    plt.imshow(reconstructed, cmap='gray', interpolation='bicubic')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='prepared.png')
    args = parser.parse_args()
    main(args.image)
