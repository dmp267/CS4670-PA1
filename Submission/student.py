import math
import numpy as np
import PIL
from matplotlib import pyplot as plt
from PIL import Image


def read_image(image_path):
    """Read an image into a numpy array.

    Args:
        image_path: Path to the image file.

    Returns:
        Numpy array containing the image
    """
    img = Image.open(image_path)
    return np.array(img)


def write_image(image, out_path):
    """Writes a numpy array as an image file.

    Args:
        image: Numpy array containing image to write
        out_path: Path for the output image
    """
    img = Image.fromarray(image)
    img.save(out_path)


def display_image(image):
    """Displays a grayscale image using matplotlib.

    Args:
        image: HxW Numpy array containing image to display.
    """
    plt.imshow(image, cmap="gray")


def convert_to_grayscale(image):
    """Convert an RGB image to grayscale.

    Args:
        image: HxWx3 uint8-type Numpy array containing the RGB image to convert.

    Returns:
        uint8-type Numpy array containing the image in grayscale
    """
    H, W = image.shape[0], image.shape[1]
    grarray = np.zeros((H, W), dtype='uint8')
    for row in range(H):
        for col in range(W):
            pixel = image[row][col]
            R, G, B = pixel[0], pixel[1], pixel[2]
            L = (299/1000) * R + (587/1000) * G + (114/1000) * B
            grarray[row][col] = int(L)
    return grarray


def convert_to_float(image):
    """Convert an image from 8-bit integer to 64-bit float format

    Args:
        image: Integer-valued numpy array with values in [0, 255]
    Returns:
        Float-valued numpy array with values in [0, 1]
    """
    return image.astype('float64') / 255.0


def convolution(image, kernel):
    """Convolves image with kernel.

    The image should be zero-padded so that the input and output image sizes
    are equal.
    Args:
        image: HxW Numpy array, the grayscale image to convolve
        kernel: hxw numpy array
    Returns:
        image after performing convolution
    """
    H, W = image.shape
    h, w = kernel.shape
    k = (h - 1) // 2  # 1/2 height
    l = (w - 1) // 2  # 1/2 side
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    convd = np.zeros((H, W))
    padded = np.pad(image, max(k, l), mode='constant', constant_values=0)
    for row in range(H):
        for col in range(W):
            local = padded[row:row+w, col:col+h]
            convd[row][col] = np.sum(local * kernel)
    return convd


def gaussian_blur(image, ksize=3, sigma=1.0):
    """Blurs image by convolving it with a gaussian kernel.

    Args:
        image: HxW Numpy array, the grayscale image to blur
        ksize: size of the gaussian kernel
        sigma: variance for generating the gaussian kernel

    Returns:
        The blurred image
    """
    kernel = np.zeros((ksize, ksize))
    acc = 0
    radius = (ksize - 1) // 2
    for row in range(-radius, radius+1):
        for col in range(-radius, radius+1):
            val = math.exp(-(row**2+col**2)/(2*(sigma**2)))
            acc += val
            kernel[row+radius][col+radius] = val

    return convolution(image, kernel/acc)


def sobel_filter(image):
    """Detects image edges using the sobel filter.

    The sobel filter uses two kernels to compute the vertical and horizontal
    gradients of the image. The two kernels are:
    G_x = [-1 0 1]      G_y = [-1 -2 -1]
          [-2 0 2]            [ 0  0  0]
          [-1 0 1]            [ 1  2  1]

    After computing the two gradients, the image edges can be obtained by
    computing the gradient magnitude.

    Args:
        image: HxW Numpy array, the grayscale image
    Returns:
        HxW Numpy array from applying the sobel filter to image
    """
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    x = convolution(image, G_x)
    y = convolution(image, G_y)

    return np.sqrt(np.square(x) + np.square(y))


def dog(image, ksize1=5, sigma1=1.0, ksize2=9, sigma2=2.0):
    """Detects image edges using the difference of gaussians algorithm

    Args:
        image: HxW Numpy array, the grayscale image
        ksize1: size of the first gaussian kernel
        sigma1: variance of the first gaussian kernel
        ksize2: size of the second gaussian kernel
        sigma2: variance of the second gaussian kernel
    Returns:
        HxW Numpy array from applying difference of gaussians to image
    """
    return gaussian_blur(image, ksize1, sigma1) - gaussian_blur(image, ksize2, sigma2)


def dft(image):
    """Computes the discrete fourier transform of image

    This function should return the same result as
    np.fft.fftshift(np.fft.fft2(image)). You may assume that
    image dimensions will always be even.

    Args:
        image: HxW Numpy array, the grayscale image
    Returns:
        HxW complex Numpy array, the fourier transform of the image
    """
    image = image.astype(np.complex128)
    H, W = image.shape
    F = np.zeros((H, W), dtype=np.complex128)
    for k in range(-H//2, H//2):
        for l in range(-W//2, W//2):
            for x in range(H):
                for y in range(W):
                    F[k+H//2][l+W//2] += image[x][y] * \
                        math.e**(-1.0j*2.0*math.pi*k*x/H-1.0j*2*math.pi*l*y/W)
    return F


def idft(ft_image):
    """Computes the inverse discrete fourier transform of ft_image.

    For this assignment, the complex component of the output should be ignored.
    The returned array should NOT be complex. The real component should be
    the same result as np.fft.ifft2(np.fft.ifftshift(ft_image)). You
    may assume that image dimensions will always be even.

    Args:
        ft_image: HxW complex Numpy array, a fourier image
    Returns:
        NxW float Numpy array, the inverse fourier transform
    """
    H, W = ft_image.shape
    f = np.zeros((H, W), dtype='float64')
    for x in range(H):
        for y in range(W):
            for k in range(-H//2, H//2):
                for l in range(-W//2, W//2):
                    f[x][y] += ft_image[k+H//2][l+W//2] * \
                        math.e**(1.0j*2.0*math.pi*k*x/H+1.0j*2*math.pi*l*y/W)
    return f


def visualize_kernels():
    """Visualizes your implemented kernels.

    This function should read example.png, convert it to grayscale and float-type,
    and run the functions gaussian_blur, sobel_filter, and dog over it. For each function,
    visualize the result and save it as example_{function_name}.png e.g. example_dog.png.
    This function does not need to return anything.
    """
    example = convert_to_float(read_image("example.png"))

    gauss = gaussian_blur(example)
    display_image(gauss)
    gauss -= gauss.min()
    write_image((gauss/gauss.max()*255.0).astype('uint8'),
                "example_gaussian_blur.png")

    sobel = sobel_filter(example)
    display_image(sobel)
    sobel -= sobel.min()
    write_image((sobel/sobel.max()*255.0).astype('uint8'),
                "example_sobel_filter.png")

    dg = dog(example)
    display_image(dg)
    dg -= dg.min()
    write_image((dg/dg.max()*255.0).astype('uint8'), "example_dog.png")


def visualize_dft():
    """Visualizes the discrete fourier transform.

    This function should read example.png, convert it to grayscale and float-type,
    and run dft on it. Try masking out parts of the fourier transform image and
    recovering the original image using idft. Can you create a blurry version
    of the original image? Visualize the blurry image and save it as example_blurry.png.
    This function does not need to return anything.
    """
    example_small = convert_to_float(read_image("example_small.png"))
    (H, W) = example_small.shape

    ksize = 5  # odd
    sigma = 1
    radius = (ksize-1)//2
    # generate gaussian kernel
    kernel = np.zeros((ksize, ksize))
    acc = 0
    for row in range(-radius, radius+1):
        for col in range(-radius, radius+1):
            val = math.exp(-(row**2+col**2)/(2*(sigma**2)))
            acc += val
            kernel[row+radius][col+radius] = val

    kernel = np.pad(kernel/acc, [(0, H), (0, W)], mode='constant')
    example_small = np.pad(
        example_small, [(0, ksize), (0, ksize)], mode='constant')

    img = idft(dft(kernel) * dft(example_small))

    img = img[radius:H+radius, radius:W+radius]
    img -= img.min()
    write_image((img/img.max()*255.0).astype('uint8'), "example_blurry.png")
