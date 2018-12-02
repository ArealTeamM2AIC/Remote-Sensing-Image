import load_data
import numpy as np
from PIL import Image
from os.path import join
from os import makedirs

# Sample
# train = load_data.load_images_in_path5("RESISC13/train/", 65)
# valid = load_data.load_images_in_path5("RESISC13/val/", 13)
# test = load_data.load_images_in_path5("RESISC13/test/", 0)

# res_directory = "starting_kit/sample_data/"

# Full data
train = load_data.load_images_in_path5("RESISC13/train/", 5200)
valid = load_data.load_images_in_path5("RESISC13/val/", 1950)
test = load_data.load_images_in_path5("RESISC13/test/", 1950)

res_directory = "starting_kit/fulldata_autoML/"

def get_classes(dataset):
    res = dict()
    labs = []
    for _, lab in dataset.values():
        labs.append(lab)
    for i, lab in enumerate(np.unique(labs)):
        res[lab] = i
    return res

def dict_toshuffledlist(d):
    l = list(d.values())
    np.random.shuffle(l)
    return l

classes = get_classes(train)
# print(classes)
train = dict_toshuffledlist(train)
valid = dict_toshuffledlist(valid)
test = dict_toshuffledlist(test)

file = open(res_directory + "Areal_train.data","w")
for path_img, _ in train:
    str_train = ""
    img = Image.open(path_img)
    arr = np.array(img).reshape(-1)
    img.close()
    for i, val in enumerate(arr):
        str_train += str(val) + " "
    str_train = str_train[:-1]
    file.write(str_train+"\n")
file.close()

file = open(res_directory + "Areal_train.solution","w")
ind = 0
for _, lab in train:
    file.write(str(classes[lab])+"\n")
file.close()

file = open(res_directory + "Areal_valid.data","w")
for path_img, _ in valid:
    str_train = ""
    img = Image.open(path_img)
    arr = np.asarray(img).reshape(-1)
    img.close()
    for i, val in enumerate(arr):
        str_train += str(val) + " "
    str_train = str_train[:-1]
    file.write(str_train+"\n")
file.close()

file = open(res_directory + "Areal_valid.solution","w")
for _, lab in valid:
    file.write(str(classes[lab])+"\n")
file.close()

file = open(res_directory + "Areal_test.data","w")
for path_img, _ in test:
    str_train = ""
    img = Image.open(path_img)
    arr = np.asarray(img).reshape(-1)
    img.close()
    for i, val in enumerate(arr):
        str_train += str(val) + " "
    str_train = str_train[:-1]
    file.write(str_train+"\n")
file.close()

file = open(res_directory + "Areal_test.solution","w")
for _, lab in test: 
    file.write(str(classes[lab])+"\n")
file.close()

file = open(res_directory + "Areal_label.name","w")
for lab in classes.keys():
    file.write("{}\n".format(lab))
file.close()

file = open(res_directory + "Areal_feat.name","w")
for i in range(1,257):
    for j in range (1, 257):
        for k in ["R", "G", "B"]:
            file.write("pixel_{}_{}_{}".format(i,j,k))
            if k != "B" or i != 256 or j != 256:
                file.write("\n")
file.close()

file = open(res_directory + "Areal_feat.type","w")
for i in range(256):
    for j in range (256):
        for k in ["R", "G", "B"]:
            file.write("int")
            if k != "B" or i != 255 or j != 255:
                file.write("\n")
file.close()
