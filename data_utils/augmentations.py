from PIL import Image
import random
import io

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