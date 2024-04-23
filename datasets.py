import torch
from torch.utils.data import Dataset
import h5py
import json
import os

from colorama import init, Fore
init()


class CaptionDataset(Dataset):
    """
    一个 PyTorch 数据集类，用于在 PyTorch DataLoader 中创建批次。
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: 存储数据文件的文件夹
        :param data_name: 处理数据集的基本名称
        :param split: 数据集分割，可以是 'TRAIN'、'VAL' 或 'TEST' 中的一个
        :param transform: 图像转换
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # 打开存储图像的 hdf5 文件
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']  # 图像数据集

        # 每张图像对应的标题数
        self.cpi = self.h.attrs['captions_per_image']

        # 加载编码后的标题
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # 加载标题长度
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch 图像转换（规范化）
        self.transform = transform

        # 总数据点数
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        """
        :param i: 第 N 个标题 enc_captions[i]
        :return:（返回第 i 个图像、标题和标题长度）。
        """

        # 第 N 个标题对应于第 (N // captions_per_image) 个图像
        # 将像素值归一化到范围[0, 1] 像素值为0~255
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            """
            如果数据集的划分不是训练集，即验证集或测试集，它会额外返回所有与该图像关联的多个标题（all_captions）。
            这对于计算 BLEU-4 分数（一种用于评估生成文本质量的指标）很有用。
            在这里，all_captions 是一个包含多个标题的张量。
            """
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        # 返回数据集的大小
        return self.dataset_size
