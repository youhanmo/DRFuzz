import itertools

import src.image_transforms as image_transforms
from params.parameters import Parameters

drfuzz = Parameters()

drfuzz.K = 64
drfuzz.batch1 = 1
drfuzz.batch2 = 1
drfuzz.p_min = 0.0  # 0.3
drfuzz.gamma = 5
drfuzz.alpha = 0.02
drfuzz.beta = 0.20
drfuzz.TRY_NUM = 50
drfuzz.MIN_FAILURE_SCORE = -100
drfuzz.framework_name = 'drfuzz'

translation = list(itertools.product([getattr(image_transforms, "image_translation")], list(range(-3, 3))))
scale = list(itertools.product([getattr(image_transforms, "image_scale")], [i * 0.1 for i in range(7, 12)]))
shear = list(itertools.product([getattr(image_transforms, "image_shear")], [0.1 * k for k in range(-6, 6)]))
rotation = list(itertools.product([getattr(image_transforms, "image_rotation")], list(range(-50, 50))))
contrast = list(itertools.product([getattr(image_transforms, "image_contrast")], [i * 0.1 for i in range(5, 13)]))
brightness = list(itertools.product([getattr(image_transforms, "image_brightness")], list(range(-20, 20))))
blur = list(itertools.product([getattr(image_transforms, "image_blur")], [k + 1 for k in range(9)]))
noise = list(itertools.product([getattr(image_transforms, "image_noise")], [1, 2, 3]))

drfuzz.G = translation + rotation + scale + shear
drfuzz.P = contrast + brightness + blur + noise
print(drfuzz.G + drfuzz.P)

drfuzz.save_batch = False
