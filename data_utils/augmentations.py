from PIL import ImageFilter
import random

class ApplyDifferentTransforms:
    def __init__(self, convert_transform, augment_transform):
        self.t1 = convert_transform
        self.t2 = augment_transform

    def __call__(self, x):
        # return the original as the first element
        q = self.t1(x)
        k = self.t2(x)
        return [q, k]
