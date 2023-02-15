import os
# import cv2
import numpy as np
import paddle

from paddle.io import Dataset
import re
from textwrap import dedent


def generate_data(path):
    f = open(path)
    text_data = f.readlines()
    list_data = []
    for frame in text_data:
        l = list(map(float, re.sub(" +", " ", dedent(frame).strip("\n")).split(" ")))
        list_data.append(l)
    return np.array(list_data)

class JointCtrl(Dataset):
    def __init__(self, joint_path_list, ctrl_path_list, transform=None, value_0 = None):
        super(JointCtrl, self).__init__()
        
        self.data_list = []
        self.label_list = []
        for joint_path, ctrl_path in zip(joint_path_list, ctrl_path_list):
            d = np.load(joint_path)
            if value_0 is not None:
                d = d - value_0
            self.data_list.extend(d.astype(dtype=np.float32))
            self.label_list.extend(np.load(ctrl_path).astype(dtype=np.float32))
        self.transform = transform


    def __getitem__(self, index):

        jiont = self.data_list[index].reshape(-1,6)
        jiont = jiont[np.newaxis, :].transpose(2, 0, 1)
        ctrl = self.label_list[index]

        if self.transform is not None:
            jiont = self.transform(jiont)
        return jiont, ctrl


    def __len__(self):

        return len(self.data_list)


    
# class JointCtrl(Dataset):
#     def __init__(self, joint_path, ctrl_path, transform=None):
#         super(JointCtrl, self).__init__()
        
#         if joint_path.endswith(".npy"):
#             self.data_list = np.load(joint_path).astype(dtype=np.float32)
#             self.label_list = np.load(ctrl_path).astype(dtype=np.float32)
#         elif joint_path.endswith(".txt"):
#             self.data_list = generate_data(joint_path).astype(dtype=np.float32)
#             # self.data_list = paddle.to_tensor(self.data_list.reshape(self.data_list.shape[0], 1, -1, 6))
#             self.label_list = generate_data(ctrl_path).astype(dtype=np.float32)
#         self.transform = transform

#     def __getitem__(self, index):

#         jiont = self.data_list[index]
#         label = self.label_list[index][0]

#         if self.transform is not None:
#             jiont = self.transform(jiont)
#         return jiont, label

#     def __len__(self):

#         return len(self.data_list)
