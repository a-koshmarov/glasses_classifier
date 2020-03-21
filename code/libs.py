# helper functions to load data from celeb folder

from PIL import Image, ImageOps
import numpy as np
import random


###########################################################
# function for loading data from celeb folder
# function dataloader(imagespath,filename)
# input : imagespath - Path where celeb images are stored, filename - file in which labels of the celeb dataset is stored
# output : image_names, labels and image pixel data
def dataloader(imagespath, filename):
    try:
        data = np.load("../data/CelebA70K.npz")
        labels = data['labels']
        all_image_names = data['imageNames']
        imageData = data['imageData']
    except FileNotFoundError:
        linenum = 0
        eyeglass_labels = []
        eyeglass_images = []
        non_eyeglass_labels = []
        non_eyeglass_images = []

        # get all 13193 eyeglasses images from 202599 celeb dataset
        for line in open(filename):
            linenum = linenum + 1
            if (linenum == 1):
                length = int(line)
            elif (linenum > 2):
                lineData = line.split(" ")
                lineData = list(filter(None, lineData))
                lineData[-1] = lineData[-1].strip()
                if (int(lineData[16]) == 1):
                    eyeglass_labels.append(1)
                    eyeglass_images.append(lineData[0])
        print(len(eyeglass_labels))
        print(len(eyeglass_images))

        eyeglass_length = len(eyeglass_images)
        noneyeglass_length = 70000 - len(eyeglass_images)

        # get non-eyeglass images
        linenum = 0
        for line in open(filename):
            linenum = linenum + 1
            if (linenum == 1):
                length = int(line)
            elif (linenum > 2):
                lineData = line.split(" ")
                lineData = list(filter(None, lineData))
                lineData[-1] = lineData[-1].strip()
                if (int(lineData[16]) == -1):
                    if (len(non_eyeglass_labels) == noneyeglass_length):
                        break
                    non_eyeglass_labels.append(0)
                    non_eyeglass_images.append(lineData[0])

        print(len(non_eyeglass_labels))
        print(len(non_eyeglass_images))

        # concatenating all eyeglasses and non-eyeglass images and labels
        all_labels = eyeglass_labels + non_eyeglass_labels
        all_image_names = eyeglass_images + non_eyeglass_images

        print(len(all_labels))
        print(len(all_image_names))

        # Random shuffling of data
        temp = list(zip(all_labels, all_image_names))
        random.shuffle(temp)
        all_labels, all_image_names = zip(*temp)

        imageData = np.zeros((70000, 28, 28), dtype=np.uint8)
        i = 0
        for images in all_image_names:
            imagesPath = imagespath + images  # full image path
            print(i, imagesPath)
            image = Image.open(imagesPath)
            image = image.convert('L')  # convert into grayscale
            image = image.resize((28, 28), Image.BICUBIC)  # resize into 28x28 dimension
            img_array = np.asarray(image)
            imageData[i, :, :] = img_array
            i = i + 1

        all_labels = np.asarray(all_labels)
        labels = np.zeros((len(all_labels), 1), dtype=np.int)
        labels[:, 0] = all_labels
        all_image_names = np.asarray(all_image_names)
        print(all_image_names.shape)
        print(all_labels.shape)
        print(imageData.shape)
        np.savez("../data/CelebA70K.npz", imageNames=all_image_names, labels=labels, imageData=imageData)
    return (imageData, labels, all_image_names)

dataloader("../data/img_align_celeba/", "../data/list_attr_celeba.txt")
