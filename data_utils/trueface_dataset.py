import random
import os
from pathlib import Path

import math

from torchvision import datasets, transforms
from PIL import Image

# facebook shard of dataset requires to be padded to 1024x1024 to allow correct batching
def pad_if_facebook(image):
    if "Facebook" in image.filename:
        pad_size = int((1024 - image.size[0]) / 2)
        resize_transform = transforms.Pad(pad_size)
        return resize_transform(image)
    else:
        return image

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
            exclude = set(['code', 'tmp', 'dataStyleGAN2'])
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
        
        self.samples = rand_generator.sample(self.samples, len(self.samples))

    def split_into_train_val(self,val_proportion=0.2):
        val_split_size = int(len(self.samples)*val_proportion)
        val_dataset = DatasetFromSamples(self.samples[0:val_split_size],self.transform)
        train_dataset = DatasetFromSamples(self.samples[val_split_size:],self.transform)
        return train_dataset,val_dataset

    def __len__(self):
        return len(self.samples)

    def find_classes(self, directory):
        classes_mapping = {"real": 0, "generated": 1}
        return self.classes, classes_mapping

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = pad_if_facebook(sample)
            sample = self.transform(sample)
        return sample, target, path

    def get_train_and_test_splits(self,split_seed):
        total_samples = len(self.samples)
        if total_samples == 0:
            print("Error! No samples present in dataset")
            return

        extractor = random.Random(split_seed)
        train_size = int(total_samples * 0.9)

        train_set = extractor.sample(self.samples,train_size)
        test_set = list(set(self.samples) - set(train_set))

        return DatasetFromSamples(train_set), DatasetFromSamples(test_set)

class DatasetFromSamples(TruefaceTotal):
    def __init__(self,samples,transform=None):
        self.classes = ["real","generated"]
        self.samples = samples
        self.transform = transform

class DatasetFromFile(TruefaceTotal):
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

class PreAndPostDataset(datasets.DatasetFolder):
    def __init__(self, socials_to_use, transform=None,real_images_amount=50,fake_images_amount=50):
        self.classes = ["real", "generated"]
        self.samples = []
        self.transform = transform
        self.downsample_fake_samples = 0
        self.real_images_count = 0
        self.fake_images_count = 0
        self.fake_images = []
        self.real_images = []
        
        presocial_base_path = "/media/mmlab/Volume/truebees/TrueFace/Train/TrueFace_PreSocial/"

        rand_generator = random.Random(451)

        for social in socials_to_use:
            # walk through the directory tree, and build tuples that are constructed as (pre_soc_path, post_soc_path)
            for dirpath, dirnames, filenames in os.walk(presocial_base_path, topdown=True):
                exclude = set(['code', 'tmp', 'dataStyleGAN2','Facebook'])
                dirnames[:] = [d for d in dirnames if d not in exclude]
                if 'FFHQ' in dirpath or 'Real' in dirpath or '0_Real' in dirpath:
                    for file in sorted(filenames):
                            if int(file.removesuffix(".png")) < 13000:
                                if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")):
                                    complete_path = os.path.join(dirpath, file)
                                    postsoc_path = self.find_corresponding_postsoc(complete_path,social)
                                    item = complete_path, postsoc_path , 0
                                    self.real_images.append(item)
                                    self.real_images_count += 1
                elif ((('StyleGAN' in dirpath or 'StyleGAN2' in dirpath or 'StyleGAN1' in dirpath)) or 'Fake' in dirpath or '1_Fake' in dirpath):
                    files_in_folder = []
                    for file in sorted(filenames):
                        if int(file.removesuffix(".png").removeprefix("seed")) < 2250:
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and (self.downsample_fake_samples % 1 == 0):
                                complete_path = os.path.join(dirpath, file)
                                postsoc_path = self.find_corresponding_postsoc(complete_path,social)
                                item = complete_path, postsoc_path, 1
                                files_in_folder.append(item)
                                self.fake_images_count += 1
                            self.downsample_fake_samples += 1
                    self.fake_images.extend(files_in_folder)   

        self.real_images = rand_generator.sample(self.real_images,real_images_amount)
        self.fake_images = rand_generator.sample(self.fake_images,fake_images_amount)

        self.samples = self.real_images + self.fake_images
    
    def __len__(self):
        return len(self.samples)

    def find_classes(self, directory):
        classes_mapping = {"real": 0, "generated": 1}
        return self.classes, classes_mapping
    
    def find_corresponding_postsoc(self,presoc_path,social_string):
        replacement = "TrueFace_PostSocial/{}".format(social_string)
        social_suffix_dict ={
            "telegram":"TL",
            "facebook":"FB",
            "whatsapp":"WA",
            "twitter":"TW",
        }

        social_suffix = social_suffix_dict[social_string.lower()]

        zero_fill = "0" if len(presoc_path.split("/")[-1].rstrip(".png").lstrip("seed")) == 5 else "00"

        postsoc_path =  presoc_path.replace("TrueFace_PreSocial",replacement)
        postsoc_path =  postsoc_path.replace("Real","0_Real")
        postsoc_path =  postsoc_path.replace("Fake","1_Fake")
        postsoc_path = postsoc_path.replace("seed",zero_fill)
        if social_string.lower() == 'twitter' or social_string.lower() == "facebook":
            postsoc_path =  postsoc_path.replace(".png","_{}.jpg".format(social_suffix))
        else:
            postsoc_path =  postsoc_path.replace(".png","_{}.jpeg".format(social_suffix))

        if "Telegram" in postsoc_path or "Facebook" in postsoc_path or "Twitter" in postsoc_path:
            postsoc_path = postsoc_path.replace("StyleGAN1","StyleGAN")

        return postsoc_path

    # TODO: change the getitem since we have to get two images in a tuple
    def __getitem__(self, index):
        presoc_path,postsoc_path, target = self.samples[index]
        presoc_sample = Image.open(presoc_path)
        postsoc_sample = Image.open(postsoc_path)
        if self.transform is not None:
            presoc_sample = pad_if_facebook(presoc_sample)
            postsoc_sample = pad_if_facebook(postsoc_sample)
            presoc_sample = self.transform(presoc_sample)
            postsoc_sample = self.transform(postsoc_sample)
            
        return (presoc_sample, postsoc_sample), target

    def split_into_train_val(self,val_proportion):
        val_split_size = int(len(self.samples)*val_proportion)
        val_dataset = PrePostFromSamples(self.samples[0:val_split_size],self.transform)
        train_dataset = PrePostFromSamples(self.samples[val_split_size:],self.transform)
        return train_dataset,val_dataset

class PrePostFromSamples(PreAndPostDataset):
    def __init__(self,samples_list,transform):
        self.samples = samples_list
        self.transform = transform

def get_dataset_size_string(ds_size):
    ds_size_string = ""
    if math.log10(ds_size) > 3:
        ds_size = math.trunc(ds_size / 1000)
        ds_size_string  = str(ds_size) + "K"
    else:
        ds_size_string = str(ds_size)
    return ds_size_string

"""        
def generateRandomSubsetToFile(dataset_path,size,dest_path):
    total_database = TuningDatabase(dataset_path)
    random_subset_samples = random.sample(total_database.samples,size)
    with open(dest_path,"w") as dest_file:
        for el in random_subset_samples:
            dest_file.write(el[0] + "\n")
"""