import random

import src.image_transforms as Image_transforms
from params.parameters import Parameters

random.seed(2021)
drfuzz = Parameters()

drfuzz.K = 64
drfuzz.batch1 = 1
drfuzz.batch2 = 1
drfuzz.p_min = 0.0
drfuzz.gamma = 5
drfuzz.alpha = 0.02
drfuzz.beta = 0.20
drfuzz.TRY_NUM = 50


def image_translation(img):
    parameters_list = list(range(-3, 3))
    x = random.choice(parameters_list)
    return Image_transforms.image_translation(img, (x, x))


def image_scale(img):
    parameters_list = [i * 0.1 for i in range(7, 12)]
    return Image_transforms.image_scale(img, random.choice(parameters_list))


def image_shear(img):
    parameters_list = [0.1 * k for k in range(-6, 6)]
    return Image_transforms.image_shear(img, random.choice(parameters_list))


def image_rotation(img):
    parameters_list = list(range(-50, 50))
    return Image_transforms.image_rotation(img, random.choice(parameters_list))


def image_contrast(img):
    parameters_list = [i * 0.1 for i in range(5, 13)]
    return Image_transforms.image_contrast(img, random.choice(parameters_list))


def image_brightness(img):
    parameters_list = list(range(-20, 20))
    return Image_transforms.image_brightness(img, random.choice(parameters_list))


def image_blur(img):
    parameters_list = [k + 1 for k in range(9)]
    return Image_transforms.image_blur(img, random.choice(parameters_list))


def image_noise(img):
    parameters_list = [1, 2, 3]
    return Image_transforms.image_noise(img, random.choice(parameters_list))


def image_erode(img):
    parameters_list = [(2, 2), (4, 4), (3, 3)]
    return Image_transforms.image_erode(img, random.choice(parameters_list))


def image_dilate(img):
    parameters_list = [(2, 2), (4, 4), (3, 3)]
    return Image_transforms.image_dilate(img, random.choice(parameters_list))


def image_reverse_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.image_dilate(img, random.choice(parameters_list))


def image_white_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.image_dilate(img, random.choice(parameters_list))


def image_black_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.image_dilate(img, random.choice(parameters_list))


def get_mutation_ops_name():
    return ['translation', 'scale', 'shear', 'rotation', 'contrast', 'brightness', 'blur', 'noise', 'erode', 'dilate', \
            'reverse_patch', 'white_patch', 'black_patch']


def get_mutation_func(name):
    if name == 'translation':
        return image_translation
    elif name == 'scale':
        return image_scale
    elif name == 'shear':
        return image_shear
    elif name == 'rotation':
        return image_rotation
    elif name == 'contrast':
        return image_contrast
    elif name == 'brightness':
        return image_brightness
    elif name == 'blur':
        return image_blur
    elif name == 'noise':
        return image_noise
    elif name == 'erode':
        return image_erode
    elif name == 'dilate':
        return image_dilate
    elif name == 'reverse_patch':
        return image_reverse_patch
    elif name == 'black_patch':
        return image_black_patch
    elif name == 'white_patch':
        return image_white_patch
