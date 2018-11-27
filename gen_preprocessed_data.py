#!/usr/bin/env python
import torch
import torch.nn as nn
import os
import torchvision
import numpy as np
from torchvision import transforms
import pickle

model_conv = torchvision.models.alexnet(pretrained=True)

n_class = 13
# Number of filters in the bottleneck layer
num_ftrs = model_conv.classifier[6].in_features
# convert all the layers to list and remove the last one
features = list(model_conv.classifier.children())[:-1]
## convert it into container and add it to our model class.
model_conv.classifier = nn.Sequential(*features)

data_dir = "RESISC13"
input_shape = 224
batch_size = 32
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
scale = 360
input_shape = 224

folders = ['train', 'val', 'test']

data_transforms = transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                      data_transforms) for x in folders}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                         shuffle=True, num_workers=4) for x in folders}
dataset_sizes = {x: len(image_datasets[x]) for x in folders}
class_names = image_datasets['train'].classes
print(class_names)


processed_data_file_name = "processed_data.data"
label_file_name = "processed_data.solution"
dico_file_name = "dico.p"

from matplotlib.pyplot import imshow
from PIL import Image

dico = {n:i for i, n in enumerate(class_names)}

pickle.dump(dico, open(dico_file_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

ft_file = open(processed_data_file_name, "w")
lbl_file = open(label_file_name, "w")

model_conv.cuda()

def write_line(file, tensor_1d):
    line = str(tensor_1d[0].item())
    for f in tensor_1d[1:-1]:
        line += " " + str(f.item())
    file.write(line + "\n")

def write_label(file, l):
    file.write(str(l.item()) + "\n")

for xb, yb in dataloaders["train"]:
    outs = model_conv(xb.cuda())
    for out, y in zip(outs.cpu(), yb):
        write_line(ft_file, out)
        write_label(lbl_file, y)
print("train fait")

for xb, yb in dataloaders["val"]:
    outs = model_conv(xb.cuda())
    for out, y in zip(outs.cpu(), yb):
        write_line(ft_file, out)
        write_label(lbl_file, y)
print("val fait")

for xb, yb in dataloaders["test"]:
    outs = model_conv(xb.cuda())
    for out, y in zip(outs.cpu(), yb):
        write_line(ft_file, out)
        write_label(lbl_file, y)
print("test fait")
