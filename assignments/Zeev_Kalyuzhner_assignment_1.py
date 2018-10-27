import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

'''
Assignment 1 - 22913 - Oct 2018
By Zeev Kalyuzhner

* exercises 3 & 4 in PDF
'''


def ex_1_a():
    # 6.5)
    rgb_im = np.zeros((512, 512, 3), 'uint8')
    r = 0.5
    g = 1
    b = 0.5

    rgb_im[..., 0] = r * 256  # R
    rgb_im[..., 1] = g * 256  # G
    rgb_im[..., 2] = b * 256  # B

    plt.imshow(rgb_im)
    plt.show(block=True)
    plt.interactive(False)


def ex_1_b():
    # 6.7)
    rgb_im = np.zeros((512, 512, 3), 'uint8')
    r = 1
    g = 1
    b = 1

    rgb_im[..., 0] = r * 200  # R
    rgb_im[..., 1] = g * 200  # G
    rgb_im[..., 2] = b * 200  # B

    plt.imshow(rgb_im)
    plt.show(block=True)
    plt.interactive(False)


def ex_2_a():
    im = misc.imread('../data/cameraman.tiff')

    def negative_img(img):
        neg_img = []
        for i in range(1, img.shape[0] - 1):
            neg_img_row = []
            for j in range(1, img.shape[1] - 1):
                neg_img_row.append(255 - img[i][j])
            neg_img.append(neg_img_row)
        return np.array(neg_img)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(im)
    axarr[1].imshow(negative_img(im))
    plt.show(block=True)
    plt.interactive(False)


def ex_2_b():
    def error_diffusion(im, t):
        for i in range(0, im.shape[0] - 1):
            for j in range(0, im.shape[1] - 1):
                old_pixel = im[i][j]
                im[i][j] = 255 if im[i][j] > t else 0
                error = old_pixel - im[i][j]
                im[i][j + 1] = im[i][j + 1] + (3 * error) // 8
                im[i + 1][j + 1] = im[i + 1][j + 1] + error // 4
                im[i + 1][j] = im[i + 1][j] + (3 * error) // 8
        return im

    img = misc.imread('../data/cameraman.tiff')
    im_ed1 = error_diffusion(img, 155)
    im_ed2 = error_diffusion(im_ed1, 155)

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(img)
    axarr[1].imshow(im_ed1)
    axarr[2].imshow(im_ed2)
    plt.show(block=True)
    plt.interactive(False)


if __name__ == '__main__':
    ex_1_a()
    ex_1_b()
    ex_2_a()
    ex_2_b()
