import csv
import cv2
import json
import os
import time
from collections import Counter
from random import seed, choice, sample
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from colorama import init, Fore

init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 将时间字符串转换为秒
def time_to_seconds(train_time):
    hours, minutes, seconds = map(float, train_time.split(':'))
    return hours * 3600 + minutes * 60 + seconds


# 将秒转换为时间字符串
def seconds_to_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def record_trian_time(train_time, elapsed_time_seconds):
    # 将时间字符串转换为秒，并相加
    total_seconds = time_to_seconds(train_time) + elapsed_time_seconds
    # 将相加后的秒数转换为时间字符串
    total_time = seconds_to_time(total_seconds)
    return total_time


def caps_to_hot(batch_size, targets, max_length, word_map=None):
    if not word_map:
        global Word_map
        word_map = Word_map
    one_hot = []
    for batch in range(batch_size):
        one_hot.append([])
        for j in range(max_length):
            one_hot_targets = [0] * len(word_map)
            one_hot_targets[targets[batch][j]] = 1
            one_hot[batch].append(one_hot_targets)
    return torch.tensor(one_hot, dtype=torch.float).to(device)


# score_to_caps
def to_caps(hot_list, is_all_hot=False, word_map=None):
    caps = list()
    if not word_map:
        global Word_map
        word_map = Word_map
    for i, all_h in enumerate(hot_list):
        caps.append([])
        cap = ''
        for h in all_h:
            index = h.index(max(h)) if is_all_hot else h
            # indices = [index for index, value in enumerate(h) if value == 1]
            key = next(key for key, value in word_map.items() if value == index)
            cap += key + " "
        caps[i].append(cap)

    return caps


# $
def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    该函数的目的是将嵌入张量（embedding tensor）用均匀分布的值填充。
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    # 假设 scores 和 targets 已经准备好
    # scores: torch.Size([batch, batch_max_caplen, onehot_size])
    # targets: torch.Size([batch, batch_max_caplen, onehot_size])
    # 将 one-hot 编码的 targets 转换回索引形式
    target_indices = torch.argmax(targets, dim=-1)
    # 计算预测值，获取沿着最后一维的最大值的索引
    # predictions = torch.argmax(scores, dim=-1)
    _, predictions = scores.topk(k, dim=-1)

    # 将预测值5个与目标索引进行比较
    target_indices = target_indices.unsqueeze(-1).expand_as(predictions)
    correct_predictions = torch.sum(predictions == target_indices)
    # 计算总预测数量
    total_predictions = targets.size(0) * targets.size(1)  # batch_size * batch_max_caplen
    # 计算准确率
    accuracy = correct_predictions.item() / total_predictions * 100.0
    # 返回准确率
    return accuracy


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def create_input_files(dataset, json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=25):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()  # $

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
        if len(captions) == 0:
            continue

        # path = os.path.join(image_folder, img['filepath'], img['filename']) \
        #     if dataset == 'coco' else os.path.join(image_folder, img['filename'])
        img_path = os.path.normpath(img['filepath'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(img_path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(img_path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(img_path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # $ 词映射
    # 创建一个单词列表，其中包含词频大于 min_word_freq 的单词
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # 创建一个字典，将单词映射到它们的索引（索引从1开始）
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 为所有输出文件创建基本/根名称
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    wordmap_path, _, _ = \
        path_checker(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), True, False)

    # 将词映射保存为 JSON
    with open(wordmap_path, 'w') as j:
        json.dump(word_map, j)
        print(Fore.GREEN + f"Success: WORDMAP file created ==> {wordmap_path}")

    # 为每张图像获取样本描述，并将图像保存到 HDF5 文件，描述及其长度保存到 JSON 文件
    seed(123)
    for impaths, imcaps, split in [(val_image_paths, val_image_captions, 'VAL'),
                                   (train_image_paths, train_image_captions, 'TRAIN'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'w') as h:
            # 记录我们每张图像采样的描述数量
            h.attrs['captions_per_image'] = captions_per_image

            # 在 HDF5 文件中创建数据集以存储图像
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\n正在读取 %s 图像和描述，存储到文件中...\n" % split, end="")
            time.sleep(0.01)
            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths, desc="图像处理")):
                # 检查图像是否成功加载
                if not os.path.exists(path):
                    print(Fore.YELLOW + f"Error: Unable to load image at {path}")
                    # 添加调试语句，输出图像的形状和数据类型
                    return

                # 采样描述
                if len(imcaps[i]) < captions_per_image:
                    # 如果此图像的现有描述数量少于 captions_per_image，
                    # 通过从现有描述中随机选择来采样额外的描述。
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    # 如果此图像的现有描述数量等于或大于 captions_per_image，
                    # 通过从现有列表中随机选择描述来采样描述。
                    captions = sample(imcaps[i], k=captions_per_image)

                # 断言检查
                assert len(captions) == captions_per_image

                # 使用 cv2 读取图像
                img = cv2.imread(impaths[i])
                # 如果需要，将 BGR 转换为 RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 检查图像是否为灰度图（2D），如果是，将其转换为 RGB（3D）
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # 将图像调整为 (256, 256)
                img = cv2.resize(img, (256, 256))

                # 转置维度以将通道维度放在第一个维度
                #  (height, width, channels) ==> (channels, height, width)
                img = np.transpose(img, (2, 0, 1))
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    # 编码标题
                    # 对标题中的每个单词进行编码，如果单词在 word_map 中不存在，则使用 <unk> 的索引
                    # 添加结束标记和填充 <pad>，确保总长度为 max_len
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            output_path, _, _ = \
                path_checker(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), True, False)
            # Save encoded captions and their lengths to JSON files
            with open(output_path, 'w') as j:
                json.dump(enc_captions, j)
                print(Fore.GREEN + f"Success: written to JSON file ==> {output_path}")
            output_path, _, _ = \
                path_checker(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), True, False)
            with open(output_path, 'w') as j:
                json.dump(caplens, j)
                print(Fore.GREEN + f"Success: written to JSON file ==> {output_path}")


def path_checker(path, is_file=False, is_create=True):
    path = os.path.abspath(path)
    base_name = os.path.basename(path)
    dir_path = os.path.dirname(path)

    if not os.path.exists(path):
        if is_create:
            if not os.path.isdir(path) and is_file:
                # 使用 open() 创建文件
                with open(path, 'w') as f:
                    pass
                print(Fore.GREEN + f"File created: {path}")
            else:
                os.makedirs(path, exist_ok=True)
                print(Fore.GREEN + f"Dir created: {path}")
        else:
            print(Fore.RED + str(path) + ' is not exists!')
    return path, dir_path, base_name


def create_csv(txt_path, csv_path=None):
    if not csv_path:
        dir_name = os.path.dirname(txt_path)
        base_name = os.path.basename(txt_path)
        csv_name = os.path.join(base_name, 'csv').split('.')[0]
        csv_path = os.path.join(dir_name, csv_name + '.csv')

    with open(txt_path, 'r') as txt_file, open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|')

        # 写入CSV文件头部
        csv_writer.writerow(['image_name', 'comment_number', 'comment'])

        # 逐行读取文本文件并转换为CSV格式写入
        i = -1
        for line in txt_file:
            if i == -1:
                i += 1
                continue
            image, caption = line.strip().split(',', maxsplit=1)
            csv_writer.writerow([image, i % 5, caption])
            i += 1
    print(Fore.GREEN + f'successful: ', Fore.BLUE + f'{txt_path}',
          Fore.GREEN + f'to', Fore.BLUE + f'{csv_path}')


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, temp_path, train_time):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'train_time': train_time,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '_epoch_' + str(epoch) + '.pth'
    epoch_path = os.path.join(temp_path, filename)
    best_path = os.path.join(temp_path, 'BEST_' + filename)
    file_path, _, _ = \
        path_checker(epoch_path, True, False)
    torch.save(state, file_path)
    print(Fore.GREEN + "Successful: save model")
    time.sleep(0.01)
    if is_best:
        file_path, _, _ = path_checker(best_path, True, False)
        torch.save(state, file_path)
        print(Fore.GREEN + "Successful: save BEST model")
        time.sleep(0.01)


def save_temp_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                         decoder_optimizer,
                         bleu4, temp_path, train_time,i):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'train_time': train_time,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'temp_checkpoint_' + data_name + f'_{str(i)}' + '.pth'
    epoch_path = os.path.join(temp_path, filename)
    file_path, _, _ = \
        path_checker(epoch_path, True, False)
    with open(file_path, 'wb') as f:
        torch.save(state, f)
        # f.flush()  # 强制将缓冲区中的数据写入磁盘
        # f.close()
    print(Fore.GREEN + "\nSuccessful: save temp model")
    time.sleep(0.01)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


if __name__ == '__main__':
    path = f'datasets/flickr8k/captions.txt'
    path, _, _ = path_checker(path, True, False)
    create_csv(path)
