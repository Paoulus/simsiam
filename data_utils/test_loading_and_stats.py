import argparse
from pathlib import Path

import torch
from torchvision import transforms

import trueface_dataset

parser = argparse.ArgumentParser(
    description="Script for testing the loading of the data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_path', '-c', type=str, default='config.json')
parser.add_argument('--device_to_use', '-d', type=str, default='cuda:0',
                    help='device to use for execution (values: cpu, cuda:0)')
parser.add_argument('--images','-i',action='store_true')
parser.add_argument('--dataset_samples','-ds',action='store_true')
parser.add_argument('--dataloader_samples','-dl',action='store_true')
parser.add_argument('--train_test_split','-tts',action='store_true')
parser.add_argument('--list_train_set',action='store_true')
parser.add_argument('--list_pre_and_post_set',action='store_true')
parser.add_argument("--test_and_val_split",action="store_true")
script_arguments = parser.parse_args()


input_folder = "/media/mmlab/Volume/truebees/TrueFace/Train/TrueFace_PostSocial/"

toTensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

total_dataset = trueface_dataset.TruefaceTotal(input_folder,
                              transforms.Compose([toTensor,normalize]),
                              seed=13776321,
                              real_amount=70,
                              fake_amount=70)

print("Total dataset length is: {}".format(len(total_dataset)))

total_sampler = None

total_dataloader = torch.utils.data.DataLoader(total_dataset,
                batch_size=32, 
                shuffle=(total_sampler is None),
                num_workers=8, 
                pin_memory=True, 
                sampler=total_sampler, 
                drop_last=True)

print("Total dataloader lenght is: {}".format(len(total_dataloader)))

if script_arguments.dataset_samples:
    print("Testing dataset construction and composition")
    dataset_stats_dict = {
        "telegram":0,
        "facebook":0,
        "twitter":0,
        "whatsapp":0,
        "trueface_postsocial":0,
        "trueface_presocial":0,
        "stylegan1":0,
        "stylegan/":0,
        "stylegan2":0,
        "stylegan3":0,
        "real":0,
        "fake":0,
        "images-psi-0.5":0,
        "images-psi-0.7":0,
        "conf-f-psi-0.5":0,
        "conf-f-psi-1":0,
        "conf-rotation-psi-0.5":0,
        "conf-translation-psi-0.5":0
    }

    with open("../trueface-dataset-loading-test.log","w") as output_file:
        for filename, label in total_dataset.samples:
            print("path: {}, label: {}".format(filename,label),file=output_file)

        real_labels_count = 0
        fake_labels_count = 0
        for filename, label in total_dataset.samples:
            image_file_path = Path(filename)
            file_path_tip = '/'.join(image_file_path.parts[-6:])
            print(str(file_path_tip).rstrip("\'"),file=output_file)
            #print("File path {} | Label {}".format(file_path_tip,x["label"].numpy()))
            patterns_to_match = list(dataset_stats_dict.keys())
            for pattern in patterns_to_match:
                if pattern in str(image_file_path).lower():
                    dataset_stats_dict[pattern] = dataset_stats_dict[pattern] + 1
            if label == 0:
                real_labels_count += 1
            elif label == 1:
                fake_labels_count +=1

    dataset_stats_dict["stylegan1"] = dataset_stats_dict["stylegan1"] + dataset_stats_dict["stylegan/"]
    del dataset_stats_dict["stylegan/"]

    print("Loaded files stats:")
    print("Real {}".format(dataset_stats_dict["real"]))
    print("Fake {}".format(dataset_stats_dict["fake"]))
    print("---stylegan1 {}".format(dataset_stats_dict["stylegan1"]))
    print("      images-psi-0.5 {}".format(dataset_stats_dict["images-psi-0.5"]))
    print("      images-psi-0.7 {}".format(dataset_stats_dict["images-psi-0.7"]))
    print("---stylegan2 {}".format(dataset_stats_dict["stylegan2"]))
    print("      conf-f-psi-0.5 {}".format(dataset_stats_dict["conf-f-psi-0.5"]))
    print("      conf-f-psi-1 {}".format(dataset_stats_dict["conf-f-psi-1"]))
    print("---stylegan3 {}".format(dataset_stats_dict["stylegan3"]))
    print("      conf-rotation-psi-0.5 {}".format(dataset_stats_dict["conf-rotation-psi-0.5"]))
    print("      conf-translation-psi-0.5 {}".format(dataset_stats_dict["conf-translation-psi-0.5"]))

    print("Social network subdivision:")
    for cat_to_print in ["telegram","facebook","twitter","whatsapp"]:
        print("{} : {}".format(cat_to_print,dataset_stats_dict[cat_to_print]))

    print("Real labels: {}".format(real_labels_count))
    print("Fake labels: {}".format(fake_labels_count))

if script_arguments.test_and_val_split:
    train_dataset, val_dataset = total_dataset.split_into_train_val(0.2)
    #verify that the two datasets are disjointed
    for path in train_dataset.samples:
        if path in val_dataset.samples:
            print("Train and val are not disjointed!")

if script_arguments.images:
    print("Testing images loading; will print a couple of image tensors")
    test_iter = iter(total_dataset)
    for i in range(3):
        image,label = next(test_iter)
        print(image)

if script_arguments.dataloader_samples:
    print("Testing Dataloader creation and iteration over multiple epochs")
    
    for epoch_n in range(5):
        print("Epoch {}/5".format(epoch_n))
        batches = len(total_dataloader)
        for i, (images, labels) in enumerate(total_dataloader):
            print("Batch {}/{}".format(i,batches))
            print(labels)

if script_arguments.list_train_set:
    print("Instantiating and listing the train set")

    train_ds, test_ds = total_dataset.get_train_and_test_splits(5098)
    print("Train set")
    for path in train_ds.samples:
        print(path)
    print("Test set")
    for path in test_ds.samples:
        print(path)

    #verify that the two datasets are disjointed
    for path in test_ds.samples:
        if path in train_ds.samples:
            print("Train and test are not disjointed!")

if script_arguments.list_pre_and_post_set:
    print("Listing pre and post set")

    # is different because it contains tuples of images, where the first one is the pre social
    # and the second one contains the correspondant post social picture for a given social network
    # max number of fake samples is 1800, due to how the IDs are organized. is this a problem? we have to 
    # validate this a bit better
    pre_and_post_ds = trueface_dataset.PreAndPostDataset("Telegram",None,1800,1800)
    pre_and_post_ds = trueface_dataset.PreAndPostDataset("Facebook",None,1800,1800)
    pre_and_post_ds = trueface_dataset.PreAndPostDataset("Twitter",None,1800,1800)
    pre_and_post_ds = trueface_dataset.PreAndPostDataset("Whatsapp",None,1800,1800)

    with open("../output-listing-pre-and-post.log","w") as log_file:
        for images, label in pre_and_post_ds:
            print(images)
            print("Presoc: {}".format(images[0].filename),file=log_file)
            print("Postsoc: {}".format(images[1].filename),file=log_file)