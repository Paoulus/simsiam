import argparse
import json

from torchvision import transforms

from pathlib import Path

from tuningDatabase import TuningDatabaseFromFile, TuningDatabaseWithRandomSampling

parser = argparse.ArgumentParser(
    description="Script for testing the loading of the data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_path', '-c', type=str, default='config.json')
parser.add_argument('--weights_path', '-m', type=str, default='./weights/gandetection_resnet50nodown_stylegan2.pth',
                    help='weights path of the network')
parser.add_argument('--device_to_use', '-d', type=str, default='cuda:0',
                    help='device to use for fine tuning (values: cpu, cuda:0)')
script_arguments = parser.parse_args()


config_file_fs = open(script_arguments.config_path,"r")
settings_json = json.loads(config_file_fs.read())


input_folder = settings_json["datasetPath"]


# load dataset and setup dataloaders and transforms (dataloader is used for batching)
transform_convert_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #transforms.Resize([256,256])
])

total_dataset = TuningDatabaseWithRandomSampling(input_folder,transform_convert_and_normalize,seed=13776321,
                                                    real_amount=settings_json["realSamplesAmount"],
                                                    fake_amount=settings_json["fakeSamplesAmount"])

print(len(total_dataset))

output_file = open("../verdoliva-dataset-loading.log","w")

for filename in total_dataset.samples:
    file_path_object = Path(str(filename[0]))
    file_path_tip = '/'.join(file_path_object.parts[-4:])
    print(file_path_tip,file=output_file)