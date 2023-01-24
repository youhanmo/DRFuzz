import numpy as np
import itertools
import src.image_transforms as image_transforms
from params.parameters import Parameters

deephunter = Parameters()

deephunter.K = 64
deephunter.batch1 = 1
deephunter.batch2 = 1
deephunter.p_min = 0.01
deephunter.gamma = 5
deephunter.alpha = 0.02
deephunter.beta = 0.20
deephunter.TRY_NUM = 50
deephunter.framework_name = 'deephunter'
translation = list(itertools.product([getattr(image_transforms, "image_translation")],
                                     [(-5, -5), (-5, 0), (0, -5), (0, 0), (5, 0), (0, 5), (5, 5)]))
rotation = list(
    itertools.product([getattr(image_transforms, "image_rotation")], [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]))
contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [1.2 + 0.2 * k for k in range(10)]))
brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], [10 + 10 * k for k in range(10)]))
blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(9)]))
scale = list(itertools.product([getattr(image_transforms, "image_scale")],
                               [1 + 0.05 * k for k in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]]))
noise = list(itertools.product([getattr(image_transforms, "image_noise")], [1, 2, 3]))
shear = list(itertools.product([getattr(image_transforms, "image_shear")], [-0.5 + 0.1 * k for k in range(10)]))

deephunter.G = translation + rotation + scale + shear
deephunter.P = contrast + brightness + blur + noise

deephunter.save_batch = False
