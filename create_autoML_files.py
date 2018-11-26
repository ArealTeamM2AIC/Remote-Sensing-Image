import load_data
import numpy as np
from PIL import Image
from os.path import join
from os import makedirs
import matplotlib.pyplot as plt

train = load_data.load_images_in_path5("RESISC13/train/", 65)#1950)
valid = load_data.load_images_in_path5("RESISC13/val/", 13)#1950)
test = load_data.load_images_in_path5("RESISC13/test/", 0)#5200)

res_directory = "starting_kit/sample_data/"
# res_directory = "fulldata_autoML/"

def get_classes(dataset):
    res = dict()
    labs = []
    for _, lab in dataset.values():
        labs.append(lab)
    for i, lab in enumerate(np.unique(labs)):
        res[lab] = i
    return res

classes = get_classes(train)
# print(classes)
# print(valid.values())
# print(valid.values())

file = open(res_directory + "Areal_train.data","w")
for path_img, _ in train.values():
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
for img, lab in train.values():
    file.write(str(classes[lab])+"\n")
    # if ind < 1:
    #     Image.open(img).show()
    #     print(lab, classes[lab])
    #     ind += 1
file.close()

file = open(res_directory + "Areal_valid.data","w")
for path_img, _ in valid.values():
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
for _, lab in valid.values():
    file.write(str(classes[lab])+"\n")
file.close()

file = open(res_directory + "Areal_test.data","w")
for path_img, _ in test.values():
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
for _, lab in test.values(): 
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
