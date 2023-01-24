import random

import src.image_transforms as Image_transforms
from params.parameters import Parameters

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
    parameters_list = [(i, j) for i in range(-3, 3) for j in range(-3, 3)]
    return Image_transforms.image_translation(img, random.choice(parameters_list))


def image_scale(img):
    return Image_transforms.image_scale(img, random.uniform(0.7, 1.2))


def image_shear(img):
    return Image_transforms.image_shear(img, random.uniform(-0.6, 0.6))


def image_rotation(img):
    return Image_transforms.image_rotation(img, random.uniform(-50, 50))


def image_contrast(img):
    return Image_transforms.image_contrast(img, random.uniform(0.5, 1.5))


def image_brightness(img):
    return Image_transforms.image_brightness(img, random.uniform(-20, 20))


def image_blur(img):
    parameters_list = [k + 1 for k in range(9)]
    return Image_transforms.image_blur(img, random.choice(parameters_list))


def image_noise(img):
    parameters_list = [1, 2, 3]
    return Image_transforms.image_noise(img, random.choice(parameters_list))


def image_erode(img):
    parameters_list = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (3, 3)]
    return Image_transforms.image_erode(img, random.choice(parameters_list))


def image_dilate(img):
    parameters_list = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (3, 3)]
    return Image_transforms.image_dilate(img, random.choice(parameters_list))


def image_reverse_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.reverse_color_patch(img, random.choice(parameters_list))


def image_shuffle_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.shuffle_patch(img, random.choice(parameters_list))


def image_white_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.white_patch(img, random.choice(parameters_list))


def image_black_patch(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w - 2) for j in range(img_d - 2)]
    return Image_transforms.black_patch(img, random.choice(parameters_list))


def image_black_pixel(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.black_pixel(img, random.choice(parameters_list))


def image_white_pixel(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.white_pixel(img, random.choice(parameters_list))


def image_reverse_pixel(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.reverse_pixel(img, random.choice(parameters_list))


def image_switch_pixel(img):
    img_w, img_d = img.shape[0], img.shape[1]
    parameters_list = [(i, j) for i in range(img_w) for j in range(img_d)]
    return Image_transforms.switch_pixel(img, random.sample(parameters_list, 2))


def gaussian_noise(img):
    parameters_list = [2, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    return Image_transforms.gaussian_noise(img, random.choice(parameters_list))


def sp_noise(img):
    return Image_transforms.saltpepper_noise(img, [])


def multiplicative_noise(img):
    parameters_list = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    return Image_transforms.multiplicative_noise(img, random.choice(parameters_list))


def get_mutation_ops_name():
    return ['translation', 'scale', 'shear', 'rotation', 'contrast', 'brightness', 'blur', 'erode', 'dilate',
            'reverse_patch', 'white_patch', 'black_patch', 'gauss_noise', 'multiplicative_noise', 'saltpepper_noise']


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
    elif name == 'shuffle_patch':
        return image_shuffle_patch
    elif name == 'black_pixel':
        return image_black_pixel
    elif name == 'white_pixel':
        return image_white_pixel
    elif name == 'black_pixel':
        return image_black_pixel
    elif name == 'reverse_pixel':
        return image_reverse_pixel
    elif name == 'switch_pixel':
        return image_switch_pixel
    elif name == 'gauss_noise':
        return gaussian_noise
    elif name == 'multiplicative_noise':
        return multiplicative_noise
    elif name == 'saltpepper_noise':
        return sp_noise
