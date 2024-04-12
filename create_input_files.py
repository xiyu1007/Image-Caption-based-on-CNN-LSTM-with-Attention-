import json
import os
import random
from colorama import init, Fore
import pandas as pd
from tqdm import tqdm

from utils import path_checker, create_input_files, create_csv

# 初始化 colorama
init(autoreset=True)


class DatasetConverter:
    def __init__(self, data_name, csv_path, image_folder, output_path_json,
                 split_type=None,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        将数据集转化为json文件，数据集格式参照Readme $data。
        :param csv_path: CSV文件路径
        :param image_folder: 图像文件夹路径
        :param output_json_path: 输出JSON文件路径(保存目录)
        :param train_ratio: 训练集比例，默认为0.7
        :param val_ratio: 验证集比例，默认为0.15
        :param test_ratio: 测试集比例，默认为0.15
        """
        self.split_type = split_type
        self.csv_path = csv_path
        self.image_folder = image_folder
        if not split_type:
            assert len(split_type) == len(csv_path)
        assert len(csv_path) == len(image_folder)

        self.data_name = data_name
        self.output_path_json = output_path_json
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 用于跟踪当前拆分的计数
        self.train_count = 0
        self.val_count = 0
        self.test_count = 0
        self.total_count = 0
        self.cpi = 5

    def convert_to_json(self, batch=None, record_parameters=True):
        # 初始化数据结构
        none_path = []
        data = {'images': []}
        for i, c in enumerate(self.csv_path):
            # 读取CSV文件
            if batch:
                df = pd.read_csv(c, delimiter='|', nrows=batch[i])
            else:
                df = pd.read_csv(c, delimiter='|')

            # 根据图片名称组织数据
            for image_name, group in tqdm(df.groupby('image_name'), desc="CSV-Processing"):
                filepath = os.path.join(self.image_folder[i], str(image_name))
                filepath = os.path.normpath(filepath)
                if not os.path.exists(filepath):
                    none_path.append(filepath)
                    continue
                # 随机确定当前图像的拆分类型
                if not self.split_type:
                    rand_num = random.uniform(0, 1)
                    if rand_num < self.train_ratio:
                        split_type = 'train'
                    elif rand_num < self.train_ratio + self.val_ratio:
                        split_type = 'val'
                    else:
                        split_type = 'test'
                else:
                    split_type = self.split_type[i]

                self.train_count += 1 if split_type == 'train' else 0
                self.val_count += 1 if split_type == 'val' else 0
                self.test_count += 1 if split_type == 'test' else 0
                # counters = {'train': self.train_count, 'val': self.val_count, 'test': self.test_count}
                # counters[split_type] += 1 if split_type in counters else 0

                image_info = {
                    'split': split_type,
                    'filepath': filepath,
                    'filename': image_name,
                    'sentences': [
                        {
                            'tokens': row['comment'].rstrip('.').split(),
                            'raw': row['comment'].rstrip('.')
                        }
                        for _, row in group.iterrows()
                    ]
                }
                data['images'].append(image_info)

        self.train_count *= self.cpi
        self.val_count *= self.cpi
        self.test_count *= self.cpi
        self.total_count = self.train_count + self.val_count + self.test_count
        # 打印拆分数量
        print(Fore.BLUE + f"Total count: {self.total_count}")
        print(Fore.BLUE + f"Train count: {self.train_count}")
        print(Fore.BLUE + f"Validation count: {self.val_count}")
        print(Fore.BLUE + f"Test count: {self.test_count}")

        json_path, _, _ = path_checker(os.path.join(self.output_path_json, self.data_name + '.json'), True, False)
        # 将数据写入JSON文件
        if self.output_path_json:
            try:
                with open(json_path, 'w') as json_file:
                    json.dump(data, json_file, indent=2)
                print(Fore.GREEN + f"Success: Datasets json file created ==> "f"{json_path}")
            except Exception as e:
                print(Fore.RED + f"Error while creating JSON file: {e}")

        self.write_parameters_to_json(none_path)

    def write_parameters_to_json(self, none_path,json_path='parameters'):
        json_path, _, _ = path_checker(os.path.join(self.output_path_json, self.data_name + '_' + json_path + '.json'),
                                       True, False)
        """
        将参数写入JSON文件
        :param json_path: 输出JSON文件路径
        """
        parameters = {
            'csv_path': self.csv_path,
            'image_folder': self.image_folder,
            'output_json_path': self.output_path_json,
            'total_count': self.total_count,
            'train_ratio': f'{self.train_count / self.total_count:.3f}',
            'val_ratio': f'{self.val_count / self.total_count:.3f}',
            'test_ratio': f'{self.test_count / self.total_count:.3f}',
            'train_count': self.train_count,
            'val_count': self.val_count,
            'test_count': self.test_count,
            'none_path': none_path,
        }
        try:
            with open(json_path, 'w') as json_file:
                json.dump(parameters, json_file, indent=2)
            print(Fore.GREEN + f"Success: Parameters written to JSON file ==> {json_path}")
        except Exception as e:
            print(Fore.RED + f"Error while writing parameters to JSON file: {e}")


def check_file(csv_path, image_folder, output_path_json, output_path_hdf5, output_path_model):
    for i, c in enumerate(csv_path):
        csv_path[i], _, _ = path_checker(c, is_file=True, is_create=False)
    for i, c in enumerate(image_folder):
        image_folder[i], _, _ = path_checker(c, is_file=False, is_create=False)
    output_path_json, _, _ = path_checker(output_path_json, is_file=False, is_create=True)
    output_path_hdf5, _, _ = path_checker(output_path_hdf5, is_file=False, is_create=True)
    output_path_model, _, _ = path_checker(output_path_model, is_file=False, is_create=True)

    return csv_path, image_folder, output_path_json, output_path_hdf5, output_path_model


def create_csv_to_json(dataset_name, csv_path, image_folder, output_path_json, split_type=None):
    # csv data to json
    converter = DatasetConverter(dataset_name, csv_path, image_folder, output_path_json, split_type=split_type)
    # 设置数据量
    converter.convert_to_json(None)


def data_flicker():
    # 使用示例
    # csv data to json
    # csv among heads no space <==> image_name|comment_number|comment
    dataset_name = 'flickr8k'
    csv_path = ['datasets/flickr8k/new_csv/captions.csv']
    image_folder = ['datasets/flickr8k/images']
    split_type = None

    output_path_json = 'out_data/flicker/out_json/new_json'
    output_path_hdf5 = 'out_data/flicker/out_hdf5/per_5_freq_5_maxlen_100'
    output_path_model = 'out_data/flicker/new_model'

    json_path = f'out_data/flicker/out_json/new_json/flickr8k.json'

    csv_path, image_folder, output_path_json, output_path_hdf5, output_path_model = \
        check_file(csv_path, image_folder, output_path_json, output_path_hdf5, output_path_model)

    create_csv(f'datasets/flickr8k/captions.txt', csv_path)

    create_csv_to_json(dataset_name, csv_path, image_folder, output_path_json, split_type=split_type)

    path_checker(json_path, True, False)
    # captions_per_image:5  min_word_freq: 5
    create_input_files(dataset_name, json_path, image_folder, 5, 5, output_path_hdf5, max_len=100)


def data_coco():
    # 使用示例
    # csv data to json
    # csv among heads no space <==> image_name|comment_number|comment
    dataset_name = 'coco'
    csv_path = ['datasets/COCO2014/coco_train2014.csv',
                'datasets/COCO2014/coco_val2014.csv']
    image_folder = ['datasets/COCO2014/train2014',
                    'datasets/COCO2014/val2014']
    split_type = ['train', 'val']
    per = 5
    freq = 5
    max_len = 100
    out_path = 'out_data/coco/'

    output_path_json = '{}out_json'.format(out_path)
    output_path_hdf5 = '{}out_hdf5/per_{}_freq_{}_maxlen_{}'.format(out_path, per, freq, max_len)
    output_path_model = '{}save_model'.format(out_path)

    json_path = '{}/{}.json'.format(output_path_json, dataset_name)

    csv_path, image_folder, output_path_json, output_path_hdf5, output_path_model = \
        check_file(csv_path, image_folder, output_path_json, output_path_hdf5, output_path_model)

    # create_csv_to_json(dataset_name, csv_path, image_folder, output_path_json, split_type=split_type)

    path_checker(json_path, True, False)

    # # captions_per_image:5  min_word_freq: 5
    create_input_files(dataset_name, json_path, image_folder, per, freq, output_path_hdf5, max_len)


if __name__ == '__main__':
    # data_flicker()
    data_coco()
