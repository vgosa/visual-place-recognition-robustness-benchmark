# -*- coding: utf-8 -*-

import ctypes
import os
import warnings
from io import BytesIO

import cv2
import numpy as np
import scipy.ndimage
import skimage as sk
from PIL import Image
from PIL import Image as PILImage
from numpy.random import RandomState
from pkg_resources import resource_filename
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# /////////////// Corruption Helpers ///////////////

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# # Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# # Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(randState, mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * randState.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height and width
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))

    top = (h - ch) // 2
    right = (w - cw) // 2
    img = scizoom(img[top:top + ch, right:right + cw], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_right = (img.shape[1] - w) // 2

    return img[trim_top:trim_top + h, trim_right:trim_right + w]


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////


def shot_noise(x, severity, randState):
    c = [500, 300, 100, 80, 40][severity - 1]

    x = np.array(x) / 255.
    return np.clip(randState.poisson(x * c) / float(c), 0, 1) * 255

## Object to research
def defocus_blur(x, severity, randState):
    c = [(1, 0.1), (3, 0.1), (4, 0.5), (5, 0.5), (6, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3xWxH -> WxHx3

    return np.clip(channels, 0, 1) * 255

## Object to research
def motion_blur(x, severity, randState):
    c = [(10, 2), (10, 3), (12.5, 4), (15, 4), (15, 5)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=randState.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity, randState):
    c = [np.arange(1, 1.01, 0.01),
         np.arange(1, 1.02, 0.01),
         np.arange(1, 1.04, 0.01),
         np.arange(1, 1.05, 0.01),
         np.arange(1, 1.06, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity, randState):
    c = [(0.5, 2), (0.75, 2), (1, 2), (1.5, 2), (2., 2)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    # mapsize has to be a power of 2, so take longest from width or height and find the nearest power of 2
    # Not sure if it's a big performance increase if it's hardcoded but I'll leave it like this for now
    highestSide = max(x.shape[:2])
    mapsize = 1
    while mapsize < highestSide:
        mapsize *= 2

    x += c[0] * plasma_fractal(randState, mapsize=mapsize, wibbledecay=c[1])[:x.shape[0], :x.shape[1]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity, randState):
    c = [(1, 0.2),
         (0.95, 0.3),
         (0.9, 0.35),
         (0.85, 0.4),
         (0.8, 0.45)][severity - 1]
    idx = randState.randint(5)
    filename = [resource_filename(__name__, 'frost/frost1.png'),
                resource_filename(__name__, 'frost/frost2.png'),
                resource_filename(__name__, 'frost/frost3.png'),
                resource_filename(__name__, 'frost/frost4.jpg'),
                resource_filename(__name__, 'frost/frost5.jpg'),
                resource_filename(__name__, 'frost/frost6.jpg')][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = randState.randint(0, frost.shape[0] - x.height), randState.randint(0, frost.shape[1] - x.width)
    frost = frost[x_start:x_start + x.height, y_start:y_start + x.width][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity, randState):
    c = [(0.0, 0.3, 3, 0.5, 8, 4, 1),
         (0.05, 0.3, 3, 0.5, 9, 4, 0.9),
         (0.1, 0.3, 2, 0.5, 10, 4, 0.8),
         (0.15, 0.3, 2, 0.5, 11, 4, 0.7),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = randState.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=randState.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1],
                                                                                          1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def brightness(x, severity, randState):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

## Object to research
def jpeg_compression(x, severity, randState):
    c = [30, 26, 22, 18, 15][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


## Object to research
# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity, randState):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.6, 244 * 0.1),
         (244 * 2, 244 * 0.5, 244 * 0.1),
         (244 * 2, 244 * 0.4, 244 * 0.1),
         (244 * 2, 244 * 0.3, 244 * 0.1)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + randState.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(randState.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(randState.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def rotate(x, severity, randState):
    c = [2.5, 5, 7.5, 10, 15][severity - 1]

    # 50/50 chance to rotate clockwise or counterclockwise
    c *= -1 if randState.choice([0, 1]) else 1
    return scipy.ndimage.rotate(x, c, reshape=False, mode='nearest')


def crop(x, severity, randState):
    c = [1.1, 1.2, 1.4, 1.5, 2][severity - 1]
    x = np.array(x)

    h = x.shape[0]
    w = x.shape[1]
    zh = int(h * c)
    zw = int(w * c)

    zoomed = cv2.resize(x, (zw, zh))

    hstart = int(randState.uniform(0, zh - h))
    wstart = int(randState.uniform(0, zw - w))
    return zoomed[hstart:hstart + h, wstart:wstart + w]


# /////////////// End Corruptions ///////////////


# /////////////// Corrupt function //////////////

corruption_tuple = (
    shot_noise,
    defocus_blur,
    motion_blur,
    zoom_blur,
    snow,
    frost,
    fog,
    brightness,
    elastic_transform,
    jpeg_compression,
    rotate,
    crop,
)

corruption_dict = {corr_func.__name__: corr_func for corr_func in corruption_tuple}

corruptions = [corr_func.__name__ for corr_func in corruption_tuple]


def corrupt(x, random, severity=1, corruption_name=None, corruption_number=-1):
    """
    :param x: image to corrupt; a WxHx3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """

    if corruption_name:
        x_corrupted = corruption_dict[corruption_name](Image.fromarray(x), severity, random)
    elif corruption_number != -1:
        x_corrupted = corruption_tuple[corruption_number](Image.fromarray(x), severity, random)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(x_corrupted)


if __name__ == '__main__':
    root_dir = './MSLS/train_val/cph/query/images'

    impath = '2fQ2nk1hXz6K--JZYwD99A.jpg'
    impath = '0m562AuWSBNDjokiN4sYRQ.jpg'
    impath = '2SGKADdy55TH8VVM9JMURA.jpg'

    img_name = os.path.join(root_dir, impath)

    for corruption in corruptions:
        # Image.open(img_name).save(f'./examples/{impath.split(".jpg")[0]}_{corruption}_0.jpg')
        Image.open(img_name).save(f'./examples/{corruption}_0.jpg')

        # Open the array and convert it to a NumPy array
        imageArrays = np.array(Image.open(img_name))

        for severity in range(1, 6):
            randState = RandomState(bytearray((impath + str(corruption) + str(severity)).encode()))
            # Corrupt the image
            corruptedArray = corrupt(imageArrays, randState, severity=severity, corruption_name=corruption)
            # Convert image back to PIL object
            corruptedImage = Image.fromarray(corruptedArray)

            # newFileName = f'./examples/{impath.split(".jpg")[0]}_{corruption}_{severity}.jpg'
            newFileName = f'./examples/{corruption}_{severity}.jpg'

            corruptedImage.save(newFileName)
