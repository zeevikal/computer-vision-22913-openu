import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology

'''
Assignment 2 - 22913 - Nov 2018
By Zeev Kalyuzhner

* parts of exercises 2, 3 & 4 in PDF
'''


def ex_1():
    im = np.array(Image.open('cameraman.tiff').convert('L'))

    f, ax = plt.subplots(2, 3)
    plt.gray()
    f.set_size_inches(18.5, 10.5)

    ax[0, 0].imshow(im)
    ax[0, 0].set_title('image')

    # image histogram
    imhist, bins = np.histogram(im.flatten(), 256, normed=True)
    ax[0, 1].bar(range(256), imhist)
    ax[0, 1].set_xlim([0, 255])
    ax[0, 1].set_title('histogram')

    # cumulative distribution function
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normilize
    ax[0, 2].plot(np.linspace(0, 1, 256), cdf)
    ax[0, 2].set_title('cdf')

    # linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    ax[1, 0].imshow(im2.reshape(im.shape))
    ax[1, 0].set_title('after equalization')

    # image histogram
    imhist, bins = np.histogram(im2.flatten(), 256, normed=True)
    ax[1, 1].bar(range(256), imhist)
    ax[1, 1].set_xlim([0, 255])
    ax[1, 1].set_title('histogram')

    # cumulative distribution function
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]  # normilize
    ax[1, 2].plot(np.linspace(0, 1, 256), cdf)
    ax[1, 2].set_title('cdf')

    plt.show()


def ex_3_a():
    # 3.23
    im = np.array(Image.open('cameraman.tiff').convert('L'))
    f, ax = plt.subplots(2, 3)
    plt.gray()
    f.set_size_inches(18.5, 10.5)

    # original image
    ax[0, 0].imshow(im)
    ax[0, 0].set_title('image')

    # median filter after original
    filtered_im = ndimage.median_filter(im, size=5)
    ax[0, 1].imshow(filtered_im)
    ax[0, 1].set_title('median filter after original')

    # laplace after median filter
    laplace_im = ndimage.laplace(filtered_im)
    ax[0, 2].imshow(laplace_im)
    ax[0, 2].set_title('laplace after median filter')

    # original image
    ax[1, 0].imshow(im)
    ax[1, 0].set_title('image')

    # laplace after original
    laplace_im2 = ndimage.laplace(im)
    ax[1, 1].imshow(laplace_im2)
    ax[1, 1].set_title('laplace after original')

    # median filter after laplace
    filtered_im2 = ndimage.median_filter(laplace_im2, size=5)
    ax[1, 2].imshow(filtered_im2)
    ax[1, 2].set_title('median filter after laplace')

    plt.show()


def ex_3_b():
    # 9.2 a)

    a = np.zeros((7, 7), dtype=np.int)
    a[1, 1] = 1
    a[2:4, 2:4] = 1
    a[4:6, 4:6] = 1
    print(a)
    s = np.array([[1, 0, 0],
                  [0, 1, 1],
                  [0, 1, 1]])
    a_hm = ndimage.binary_hit_or_miss(a, structure1=s).astype(np.int)
    print(a_hm)

    # b)
    # Only one pass is required. Application of the hit-or-miss transform using a given structure
    # finds all instances of occurrence of the pattern described by that structuring element.

    # c) The order does matter.

    s1 = np.array([[0, 1, 0],
                   [1, 1, 0],
                   [0, 0, 0]])
    s2 = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0]])
    s3 = np.array([[0, 0, 0],
                   [0, 1, 1],
                   [0, 1, 0]])
    s4 = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [0, 1, 0]])
    b = np.zeros((4, 10), dtype=np.int)
    b[1, 5:10] = 1
    b[2, 0:6] = 1
    # b[4:6, 4:6] = 1
    print(b)
    b_hm_s1 = ndimage.binary_hit_or_miss(b, structure1=s1).astype(np.int)
    b_hm_s2 = ndimage.binary_hit_or_miss(b, structure1=s2).astype(np.int)
    b_hm_s3 = ndimage.binary_hit_or_miss(b, structure1=s3).astype(np.int)
    b_hm_s4 = ndimage.binary_hit_or_miss(b, structure1=s4).astype(np.int)
    print(b_hm_s1, b_hm_s2, b_hm_s3, b_hm_s4)


def ex_3_c():
    # 9.5)
    scale = 16  # resolution of shape

    # draw original image
    shape = np.zeros((8 * scale, 8 * scale), dtype=np.int)
    shape[1 * scale:7 * scale, 1 * scale:3 * scale] = 1
    shape[1 * scale:7 * scale, 5 * scale:7 * scale] = 1
    shape[int(5.3 * scale):7 * scale, 1 * scale:7 * scale] = 1
    plt.imshow(shape)

    edore_a_struct = morphology.square(scale)
    edore_a_origin = (scale // 2 - 1, scale // 2 - 1)
    edore_a = ndimage.binary_erosion(shape, structure=edore_a_struct, origin=edore_a_origin)
    plt.imshow(edore_a, cmap='binary_r')

    edore_b = ndimage.binary_erosion(shape,
                                     structure=morphology.rectangle(int(5 * scale), int(.5 * scale)),
                                     origin=(int(2 * scale), 0))
    plt.imshow(edore_b)

    edore_c_struct = morphology.rectangle(scale * 2, scale * 2)
    edore_c = ndimage.binary_erosion(shape, structure=edore_c_struct)
    dilate_c_struct = morphology.disk(scale // 2)
    shape_c = ndimage.binary_dilation(edore_c, structure=dilate_c_struct)
    plt.imshow(edore_c, cmap='binary_r')

    dilate_d_struct = ndimage.binary_dilation(shape, structure=morphology.disk(scale // 2))
    edore_d_struct = ndimage.binary_erosion(dilate_d_struct, structure=morphology.disk(scale // 4))
    plt.imshow(edore_d_struct)


if __name__ == '__main__':
    ex_1()
    ex_3_a()
    ex_3_b()
    ex_3_c()
