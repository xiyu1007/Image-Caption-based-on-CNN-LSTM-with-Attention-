import os
from pycocotools.coco import COCO
import csv
import matplotlib.pyplot as plt


def coco_to_csv(annFile, csv_path):
    coco = COCO(annFile)
    # 打开CSV文件以写入模式
    with open(csv_path, 'w', newline='') as csvfile:
        # 定义CSV写入器
        writer = csv.writer(csvfile, delimiter='|')

        # 写入表头
        writer.writerow(['image_name', 'comment_number', 'comment'])

        # 获取所有图片ID
        imgIds = coco.getImgIds()

        # 遍历每张图片并获取其所有标题
        for imgId in imgIds:
            # 获取图片信息
            img = coco.loadImgs(imgId)[0]
            # 获取图片的标注ID
            annIds = coco.getAnnIds(imgIds=img['id'])
            # 获取图片的标注
            anns = coco.loadAnns(annIds)

            # 遍历每个标注并将其写入CSV文件
            for idx, ann in enumerate(anns):
                writer.writerow([img['file_name'], idx, ann['caption']])


if __name__ == '__main__':
    dataDir = 'datasets/COCO2014/annotations_trainval2014'
    dataType = 'val2014'  # or 'train2014' for training set
    annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    csv_path = 'datasets/COCO2014/coco_{}.csv'.format(dataType)
    coco_to_csv(annFile, csv_path)

    dataType = 'train2014'  # or 'train2014' for training set
    annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    csv_path = 'datasets/COCO2014/coco_{}.csv'.format(dataType)
    coco_to_csv(annFile, csv_path)
