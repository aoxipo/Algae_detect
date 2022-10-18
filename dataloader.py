import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
#from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import glob



class Dataload(Dataset):
    def __init__(self, file_path, batch_size = 1, data_source = None, gray = False, image_shape = (128,128),
                 keep_same = True, limit = None, shuffle = False,same_matrix = True):
        self.file_path = file_path
        self.data_source = data_source
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.limit = limit

        self.photo_set = {}
        self.same_matrix =same_matrix
        self.gray = gray 
        if(gray):
            self.channels = 1
        else:
            self.channels = 3
        self.load_data(file_path, keep_same, shuffle = shuffle)
        self.set_gan()
        
        # self.X_train, self.Y_train = self.load_all_data(False ,gray, "train")
        # self.X_val, self.Y_val = self.load_all_data(False ,gray, "val")
        
    def check_dir(self, path):
        if (not os.path.exists(path)):
            return 0
        return 1
    
    def read_image_data(self, file_path, gray = False):
        if(gray):
            image = cv2.imread(file_path, 0) 
        else:
            image = cv2.imread(file_path)
        if(image is None):
             raise RuntimeError('image can \'t read:' + file_path)
        return image

    def set_gan(self):
        cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
        cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)
        method_list = [
                transforms.ToPILImage(),
                transforms.Resize(self.image_shape),
                transforms.ToTensor(),
                transforms.Normalize(cifar_norm_mean, cifar_norm_std),
        ]
        self.datagen = transforms.Compose(method_list)


    def load_data(self, file_path , keep_same = True, shuffle = True):
        self.file_path_list = []
        self.label_list = []
        self.label_list_True = []
        self.label_list_False = []
        self.lable_list_val = []

        self.label_dict = {
            0:'Bacillariophyta',
            1:'Chlorella',
            2:'Chrysophyta',
            3:'Dunaliella_salina',
            4:'Platymonas',
            5:'translating_Symbiodinium',
            6:'bleaching_Symbiodinium',
            7:'normal_Symbiodinium}'
        }
        middle = file_path + "\\images\\"
        label_path = file_path + "\\labels\\"
        for i in os.listdir(middle):
            num = os.path.splitext(i)[0]
            path = []
            x = middle + num + '.png'
            y = label_path + num + '.txt'
            path.append(x)
            path.append(y)
            self.photo_set[int(num)] = path

        print(middle, label_path)
        if(not self.check_dir(middle)):
            if(self.dataset_type == "train"):
                raise RuntimeError('train dir not exists:'+ middle)
            else:
                raise RuntimeError('val dir not exists:' + middle)


            # pass

        print("total:", len(self.photo_set))

    def __iter__(self):
        return self


    def __getitem__(self, index):
        """
        获取对应index的图像，并视情况进行数据增强
        """
        if(index >= self.__len__()):
            return
        try:
            a = index +430
            if(len(self.photo_set)>0):

                image = self.read_image_data( self.photo_set[a][0] ,self.gray)
                # print(image)
                label = []
                with open(self.photo_set[a][1]) as f:
                    lines = f.readlines()
                    for line in lines:
                        x = line.replace('\n', '').split(' ')
                        x = [float(i) for i in x]
                        label.append(x)
                label = torch.tensor(label)
                l, m = label.shape
                if(self.same_matrix):
                    # print("取矩阵大小相同")
                    # 训练集最大的向量为20
                    ones = -1 * np.ones([25, 5])
                    ones[:l,:] =label
                    label = ones
                # print(ones)
                else:
                    # print("取矩阵大小不同")
                    pass


            if self.datagen is not None:
                image = self.datagen(image)

            return image, label
     

        except Exception as e:
            print("发现异常")
            print(e.__class__.__name__)
            print(e)



    def __len__(self):
        x = len(self.photo_set)
        # print(x)
        return x
    
         

if __name__ == '__main__':
    
    batch_size = 32

    train_dataloader = Dataload(r"E:\Dataset\training_set\train")

    train_loader= DataLoader(
        dataset = train_dataloader,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True
    )
    # a, b= train_loader[500]
    # print("photo")
    # print(a)
    # print("label")
    # print(b)

    for data in train_loader:
        img,label =data
        print(img.shape)