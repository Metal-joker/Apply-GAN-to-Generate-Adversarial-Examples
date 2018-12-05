import json
import os
import random
import time
from xml.dom import minidom

import matplotlib.pyplot as plt
import torch
import torchvision as tv
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets as ds
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNet_val(data.Dataset):
    """Dataset class for Clatech101 Dataset."""

    def __init__(self, image_path, xml_path, json_path, transform=None):
        """initializing and preprocessing"""
        self.image_path = image_path
        self.json_path=json_path
        self.xml_path=xml_path
        self.transform = transform
        self.dataset = []
        self.class_dict = {}
        self.preprocess()

        self.num_images = len(self.dataset)

    def preprocess(self):
        # create class dict
        file = open(self.json_path)
        self.class_dict = json.load(file)
        # get data in class 1-5
        for root, dirs, files in os.walk(self.image_path):
            for file in files:
                root_set = os.path.split(file)
            # need to be debug here
                label = get_image_label(
                    self.xml_path + root_set[1][:-5] + '.xml',self.json_path)
                if label in range(0, 5):
                    self.dataset.append([root+file, label])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        file_path, label = self.dataset[index]
        image = Image.open(file_path)
        return self.transform(image), torch.Tensor(get_onehot_list(label))

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_image_label(xml_path,json_path):
    xml_doc = minidom.parse(xml_path)
    img_name = xml_doc.getElementsByTagName('name')[0].childNodes[0].data
    file = open(json_path)
    class_dict = json.load(file)
    for key, label_list in class_dict.items():
        if label_list[0] == img_name:
            return int(key)

def get_onehot_list(label):
    list_foo = [0, 0, 0, 0, 0]
    list_foo[label] = 1
    return list_foo
