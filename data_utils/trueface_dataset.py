import random
import os

from torchvision import datasets
from PIL import Image

class TruefaceTotal(datasets.DatasetFolder):
    def __init__(self, path,transform = None,real_amount=50,fake_amount=50,seed=451):
        exclude = set(['code','tmp','dataStyleGAN2'])
        self.classes = ["real", "generated"]
        self.samples = []
        self.transform = transform
        self.downsample_fake_samples = 0
        self.real_images_count = 0
        self.fake_images_count = 0
        self.fake_images = []
        self.real_images = []
        self.root = path

        rand_generator = random.Random(seed)

        for dirpath, dirnames, filenames in os.walk(path, topdown=True):
            exclude = set(['code', 'tmp', 'dataStyleGAN2','Facebook'])
            dirnames[:] = [d for d in dirnames if d not in exclude]
            if 'FFHQ' in dirpath or 'Real' in dirpath or '0_Real' in dirpath:
                for file in sorted(filenames):
                        if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                            item = os.path.join(dirpath, file), 0
                            self.real_images.append(item)
                            self.real_images_count += 1
            elif ((('StyleGAN' in dirpath or 'StyleGAN2' in dirpath)) or 'Fake' in dirpath or '1_Fake' in dirpath):
                files_in_folder = []
                for file in sorted(filenames):
                    if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and (self.downsample_fake_samples % 5 == 0):
                        item = os.path.join(dirpath, file), 1
                        files_in_folder.append(item)
                        self.fake_images_count += 1
                    self.downsample_fake_samples += 1
                self.fake_images.extend(files_in_folder)
        real_amount = min(real_amount,len(self.real_images))
        assert real_amount > 0, "Amount of requested real images is zero!!"
        self.real_images = rand_generator.sample(self.real_images,real_amount)

        fake_amount = min(fake_amount,len(self.fake_images))
        assert fake_amount > 0, "Amount of requested fake images is zero!!"
        self.fake_images = rand_generator.sample(self.fake_images,fake_amount)
        self.samples = self.real_images + self.fake_images
        
    def __len__(self):
        return len(self.samples)

    def find_classes(self, directory):
        classes_mapping = {"real": 0, "generated": 1}
        return self.classes, classes_mapping

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class DatabaseFromSamples(TruefaceTotal):
    def __init__(self,samples,transform=None):
        self.classes = ["real","generated"]
        self.samples = samples
        self.transform = transform

class DatabaseFromFile(TruefaceTotal):
    def __init__(self,samples_file_path,transform=None):
        self.classes = ["real","generated"]
        self.samples = []
        self.transform = transform

        with open(samples_file_path) as samples_file_desc:
            for line in samples_file_desc:
                label = 1 if ("StyleGAN" in line) or ("StyleGAN2" in line) else  0
                item = line.strip("\n"), label
                self.samples.append(item)

class FolderDataset(TruefaceTotal):
    def __init__(self,path,transform=None,label=0):
        self.samples = []
        self.classes = ["real","generated"]
        self.transform = transform
        self.root = path

        for root_1, dirs_1, files_1 in os.walk(path, topdown=True):
            for file in sorted(files_1):
                item = os.path.join(root_1,file),label
                self.samples.append(item)
"""        
def generateRandomSubsetToFile(dataset_path,size,dest_path):
    total_database = TuningDatabase(dataset_path)
    random_subset_samples = random.sample(total_database.samples,size)
    with open(dest_path,"w") as dest_file:
        for el in random_subset_samples:
            dest_file.write(el[0] + "\n")
"""