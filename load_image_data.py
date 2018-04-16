import os
import numpy as np
import cv2
from scipy import misc
import random
from tqdm import tqdm


class LoadData():
    def __init__(self, data_path):
        self.labels = []
        self.one_hot_encoded_labels = []
        self.load_images_and_data(data_path)
        self.num_labels = len(self.labels)

        self.training_data = self.format_training_data(self.labels, data_path)

        print(len(self.training_data))

    def load_images_and_data(self,data_path):
        di = os.fsencode(data_path)
        for file_name in os.listdir(di):
            self.labels.append(file_name)
            # print(self.labels)

    def format_training_data(self, labels, data_path):
        di = os.fsencode(data_path)
        images_with_labels = []

        images = []
        labels = []
        print("\n\n*<Loading Data>*")
        for label in tqdm(os.listdir(di)):
            try:
                one_hot_label = np.zeros([self.num_labels])
                one_hot_label[self.labels.index(label)] = 1

                for img in tqdm(os.listdir(data_path+"/"+(label.decode("utf-8")))):
                    try:
                        img_path= data_path+"/"+(label.decode("utf-8"))+"/"+img
                        img_arr = misc.imread(img_path)
                        r_img = misc.imresize(img_arr, [80,80])
                        images_with_labels.append([r_img,one_hot_label])
                    except OSError:
                        pass
            except NotADirectoryError:
                pass
        # print(images_with_labels)
        random.shuffle(images_with_labels)
        random.shuffle(images_with_labels)
        random.shuffle(images_with_labels)
        for il in images_with_labels:
            images.append(il[0])
            labels.append(il[1])

        return images, labels



# LoadData("../../Downloads/the-simpsons-characters-dataset/simpsons_dataset")
