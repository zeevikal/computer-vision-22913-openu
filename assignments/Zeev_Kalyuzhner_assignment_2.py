import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

'''
Assignment 2 - 22913 - Nov 2018
By Zeev Kalyuzhner

* exercises 2, 3 & 4 in PDF
'''


def ex_1():
    im = misc.imread('cameraman.tiff')
    w, h = im.shape

    def filter_img(img_path, kernel):
        img = misc.imread(img_path)
        w, h = img.shape
        filtered_img = np.zeros(img.shape)
        for i in range(1, w - 3):
            for j in range(2, h - 3):
                neigh_img = img[i - 1:i + 2, j - 1:j + 2]
                kernel_output = np.sum(neigh_img * kernel)
                filtered_img[i, j] = kernel_output
        return filtered_img

    kernel_1 = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    plt.imshow(filter_img('cameraman.tiff', kernel_1))
    plt.show(block=True)
    plt.interactive(False)

    kernel_2 = np.array([[-1, -1, -1], [-1, -10, -1], [-1, -1, -1]])
    plt.imshow(filter_img('cameraman.tiff', kernel_2))
    plt.show(block=True)
    plt.interactive(False)
    # as we can see, if our kernel will be negative the image we'll get lower values,
    # till it will be a negative image of the original one.


def ex_2_a():
    # full solution in PDF
    x = [4, 2, 3]
    h = [2, 5, 1]
    print(np.convolve(x, h))


if __name__ == '__main__':
    ex_1()
    ex_2_a()