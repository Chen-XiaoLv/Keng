"""Base augmentations operators."""

import numpy as np

from PIL import Image, ImageOps, ImageEnhance
import random
import warnings


# suppress warnings
warnings.filterwarnings('ignore')

# ImageNet code should change this value to 224
IMAGE_SIZE = 640
# IMAGE_SIZE = 224



#########################################################
#################### AUGMENTATIONS ######################
#########################################################


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


#############Aug##############################

def invert(pil_img, _):
    return ImageOps.invert(pil_img)


def mirror(pil_img, _):
    return ImageOps.mirror(pil_img)

def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)

# operation that overlaps with ImageNet-C's test set
def autocontrast(pil_img, level):
    level = float_parameter(sample_level(level), 10)
    return ImageOps.autocontrast(pil_img, 10 - level)

# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)

augmentations = [
   equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, mirror, invert
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness, mirror, invert
]

# augmentations_=[
#     translate_x,translate_y,posterize,equalize,shear_x, shear_y,rotate
# ]

augmentations_=[
    posterize,translate_y,translate_x,rotate
]
################################################################
######################## Pixels_MIXINGS ########################
################################################################

def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    out = (out + 1) / 2
    return out


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1 ** a) * (img2.clip(1e-37) ** b)
    out = out / 2
    return out


def IHS(data_low, data_high, alpha=0.7):
    Trans = np.matrix([
        [1. / 3., 1. / 3., 1. / 3.],
        [-2 ** 0.5 / 6, -2 ** 0.5 / 6, 2 * 2 ** 0.5 / 6],
        [1 / 2 ** 0.5, -1 / 2 ** 0.5, 0]
    ])

    Itrans = np.matrix([
        [1, -1 / 2 ** 0.5, 1 / 2 ** 0.5],
        [1, -1 / 2 ** 0.5, -1 / 2 ** 0.5],
        [1, 2 ** 0.5, 0]
    ])

    data_high = data_high.transpose()
    data_low = data_low.transpose()

    data_high = data_high.reshape(3, 640 * 640)
    data_low = data_low.reshape(3, 640 * 640)

    AIHS = np.dot(Trans, np.matrix(data_high))
    BIHS = np.dot(Trans, np.matrix(data_low))


    BIHS[0, :] = BIHS[0, :] * (1 - alpha) + AIHS[0, :] * (alpha)

    RGB = np.array(np.dot(Itrans, BIHS))
    RGB = RGB.reshape((3, 640, 640))

    return RGB.transpose()