from PIL import Image
import random
import io
import math

import torchvision.transforms as transforms

class ApplyDifferentTransforms:
    def __init__(self, convert_transform, augment_transform):
        self.t1 = convert_transform
        self.t2 = augment_transform

    def __call__(self, x):
        # return the original as the first element
        q = self.t1(x)
        k = self.t2(x)
        return [q, k]

class CompressToJPEG:
    def __init__(self):
        pass

    def __call__(self,x):
        compressed_buffer = io.BytesIO()
        x.save(compressed_buffer,format='JPEG')
        x = Image.open(compressed_buffer,formats=['JPEG'])
        return x
    
class CompressToJPEGWithRandomParams:
    def __init__(self):
        pass

    def __call__(self,x):
        rand_quality = random.randint(75,90)
        rand_qtables_preset = random.choice(['web_low','web_high'])
        compressed_buffer = io.BytesIO()
        x.save(compressed_buffer,format='JPEG',
                       quality=rand_quality,
                       qtables=rand_qtables_preset)
        x = Image.open(compressed_buffer,formats=['JPEG'])
        return x

class ResizeAtRandomLocationAndPad:
    def __init__(self,min,max):
        self.min = min
        self.max = max

    def __call__(self,x):
        chosen_size = random.randint(self.min,self.max - 4)
        x = transforms.RandomCrop(chosen_size)(x)
        amount_to_pad = (self.max-x.size[0]) / 2
        pad_one = math.ceil(amount_to_pad)
        pad_second = math.floor(amount_to_pad)
        #  left, top, right and bottom borders respectively.
        x = transforms.Pad((pad_one,pad_one,pad_second,pad_second))(x)
        return x