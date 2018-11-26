from PIL import Image
import re
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import shutil


def load_images_in_path(path, limit_file_per_town):
    return {f: load_one_image(join(path, f)) for f in listdir(path) if isfile(join(path, f)) and int(re.search("[0-9]+", f).group(0)) < limit_file_per_town}

def load_images_in_path2(path):
    return {f: load_one_image(join(path, f)) for i, f in enumerate(listdir(path)) if isfile(join(path, f))}

def load_images_in_path3(path, gt_path, num_toload):
    # list_path = []
    # list_path.append(listdir(path))
    # list_path.append(listdir(gt_path))
    # list_path = np.transpose(list_path)
    # np.random.shuffle(list_path)
    # list_path = np.transpose(list_path)
    tmp = list(zip(sorted(listdir(path)), sorted(listdir(gt_path))))
    random.shuffle(tmp)
    list_path, list_gtpath = zip(*tmp)
    print(list_path[0])
    print(list_gtpath[0])
    return {f: load_one_image(join(path, f)) for i, f in enumerate(list_path) if isfile(join(path, f)) and i < num_toload}, {f: load_one_image(join(gt_path, f)) for i, f in enumerate(list_gtpath) if isfile(join(gt_path, f)) and i < num_toload}

def load_images_in_path4(path, gt_path, num_train, num_valid, num_test):
    # list_path = []
    # list_path.append(listdir(path))
    # list_path.append(listdir(gt_path))
    # list_path = np.transpose(list_path)
    # np.random.shuffle(list_path)
    # list_path = np.transpose(list_path)
    tmp = list(zip(sorted(listdir(path)), sorted(listdir(gt_path))))
    random.shuffle(tmp)
    list_path, list_gtpath = zip(*tmp)
    return {f: load_one_image(join(path, f)) for i, f in enumerate(list_path) if isfile(join(path, f)) and i < num_train}, {f: load_one_image(join(gt_path, f)) for i, f in enumerate(list_gtpath) if isfile(join(gt_path, f)) and i < num_train},{f: load_one_image(join(path, f)) for i, f in enumerate(list_path) if isfile(join(path, f)) and i >= num_train and i < num_valid + num_train}, {f: load_one_image(join(gt_path, f)) for i, f in enumerate(list_gtpath) if isfile(join(gt_path, f)) and i >= num_train and i < num_valid + num_train},{f: load_one_image(join(path, f)) for i, f in enumerate(list_path) if isfile(join(path, f)) and i >= num_train + num_valid and i < num_test + num_train + num_valid}, {f: load_one_image(join(gt_path, f)) for i, f in enumerate(list_gtpath) if isfile(join(gt_path, f)) and i >= num_train + num_valid and i < num_test + num_train + num_valid}

def load_images_in_path5(path, num_toload):
    tmp = list(sorted(listdir(path)))
    list_path = tmp
    # print(list_path)
    num_classes = len(list_path)
    if num_toload % num_classes != 0:
        print("You should take a multiple of the number of classes")
        return dict()
    else:
        paths = [path + classe for classe in list_path]
        number_bypaths = int(num_toload / num_classes)
        res = dict()
        for p in paths :
            list_img = listdir(p)
            for num in range(number_bypaths):
                f = list_img[num]
                res[f] = (join(p, f), re.search("[A-Za-z]+", f).group(0))
        return res

def load_one_image(file):
    return Image.open(file)


def get_town_name_list(img_dict):
    names = [f for f,img in img_dict.items()]
    names = list(map(lambda n: re.search("[A-Za-z]+", n).group(0), names))
    return set(names)
