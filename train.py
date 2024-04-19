import os.path
import sys
import threading
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication
from nltk.translate.bleu_score import corpus_bleu
from torch import nn

from Win_Qt import MainWindow
from config import Config
from datasets import *
from models import Encoder, DecoderWithAttention
from utils import *

init()

data_folder = f'out_data/coco/out_hdf5/per_5_freq_5_maxlen_20'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
temp_path = 'out_data/coco/save_model'

# Model parameters
emb_dim = 512  # 词嵌入的维度
attention_dim = 1024  # 注意力机制中线性层的维度
decoder_dim = 512  # 解码器RNN的维度
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# （仅当模型的输入具有固定大小时设置为true；否则会有很多计算开销）
cudnn.benchmark = True

# 训练参数
start_epoch = 0  # 开始的训练轮次
epochs = 100  # 训练的总轮次
epochs_since_improvement = 0  # 自上次在验证集上取得改进以来的轮次数，用于提前停止
batch_size = 16  # 32 每个训练批次中的样本数
workers = 0  # 数据加载的工作进程数 num_workers参数设置为0，这将使得数据加载在主进程中进行，而不使用多进程。
# 这个错误是由于h5py对象无法被序列化（pickled）引起的。
# 在使用多进程（multiprocessing）加载数据时，数据加载器（DataLoader）会尝试对每个批次的数据进行序列化，以便在不同的进程中传递。
encoder_lr = 1e-4  # 编码器的学习率（如果进行微调）
decoder_lr = 2e-4  # 解码器的学习率
grad_clip = 5.  # 梯度裁剪的阈值，用于防止梯度爆炸
alpha_c = 1.  # '双重随机注意力'的正则化参数
best_bleu4 = 0.  # 当前的最佳 BLEU-4 分数
print_freq = 100  # 每训练多少个批次打印一次训练/验证统计信息
fine_tune_encoder = False  # 是否对编码器进行微调

# 检查点的路径，如果为 None，则没有检查点
train_time = "00:00:00"
start_time = 0
number = 0
window = None
log_path = os.path.join(temp_path, "save_log.json")
checkpoint = r"out_data/coco/save_model/temp_checkpoint_coco_5_cap_per_img_5_min_word_freq_0.pth"
checkpoint, _, _ = path_checker(checkpoint, True, False)
if os.path.exists(checkpoint):
    file_base = checkpoint.split('_')  # 使用下划线分割文件名
    number_index = file_base.index('freq') + 1  # 获取编号的索引位置
    number = int(file_base[number_index].split('.')[0])  # 获取编号并转换为整数


# checkpoint = None


def main():
    """
    Training and validation.
    """
    config = Config()

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name
    global word_map, train_time, start_time, log_path

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    word_map_file = os.path.normpath(word_map_file)

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None or not os.path.exists(checkpoint):
        # LSTM
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        # ResNet
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

        with open(log_path, "w") as file:
            json.dump({"save_flag": False, "train_time": "00:00:00"}, file)
    else:
        checkpoint = torch.load(checkpoint)
        train_time = checkpoint['train_time']
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)
        with open(log_path, "w") as file:
            json.dump({"save_flag": False, "train_time": train_time}, file)
        print(Fore.GREEN + 'Model loading from => \n' + str(checkpoint))
        time.sleep(1)
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 创建一个图像变换组合，包含 normalize 变换
    transform = transforms.Compose([normalize])
    # TODO num_workers = 0
    """
    shuffle=True：表示在每个 epoch 开始时是否对数据进行随机洗牌。 
    这有助于确保每个批次都包含来自不同样本的数据，有助于模型的训练。 
    num_workers：表示用于加载数据的子进程的数量。这有助于加速数据加载过程。若报错改为0
    pin_memory=True：如果设为 True，数据加载器将数据加载到 
    CUDA 的固定内存区域，以便更快地将数据传递给 GPU。这在使用 GPU 训练时可以提高性能。 
    """
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    # TODO
    for epoch in range(start_epoch, epochs):
        # 如果连续 20 个 epoch 都没有性能提升，则提前终止训练
        if epochs_since_improvement == 20:
            break
        # 如果经过一定 epoch 数（3 的倍数）仍然没有性能提升，则进行学习率衰减
        if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
            # 调整解码器（decoder）的学习率，将当前学习率乘以 0.8
            adjust_learning_rate(decoder_optimizer, 0.8)
            # 如果需要对编码器（encoder）进行微调，也调整编码器的学习率
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # 假设训练开始
        start_time = time.time()
        # # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              word_map=word_map,
              config=config)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                word_map=word_map)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        start_time = time.time()
        train_time = record_trian_time(train_time, elapsed_time_seconds)
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, temp_path, train_time=train_time)

        time.sleep(1)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map, config):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    global train_time, start_time, log_path, number, window

    # 设置解码、编码器为训练模式（启用 dropout 和批归一化）
    decoder.train()
    encoder.train()

    # 用于记录每个单词的损失的指标
    losses = AverageMeter()  # loss (per word decoded)
    # 用于记录 top-5 准确率的指标
    top5accs = AverageMeter()  # top5 accuracy

    # Batches
    with tqdm(total=len(train_loader), desc=f"Training:  Epoch {epoch}/{epochs}") as t:
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # 输出为(batch_size, encoded_image_size 14, encoded_image_size 14, 通道维度 2048)
            imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]  # torch.Size([batch, max_caption_len])

            targets = caps_to_hot(len(scores), targets, max(decode_lengths), word_map)
            targets.to(device)

            loss = criterion(scores, targets)
            # Add doubly stochastic attention regularization
            #  TODO
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))

            t.set_postfix(loss=f"{losses.val:.4f}({losses.avg:.4f})",
                          top5=f"{top5accs.val:.3f} ({top5accs.avg:.3f})")
            t.update(1)
            config.check_timeout()
            if config.save_flag:
                config.save_flag = False
                end_time = time.time()
                elapsed_time_seconds = end_time - start_time
                start_time = time.time()
                train_time = record_trian_time(train_time, elapsed_time_seconds)
                save_temp_checkpoint(data_name, epoch-1, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                                     decoder_optimizer, 0, temp_path, train_time, number + i + 1)
                with open(log_path, "w") as file:
                    json.dump({"save_flag": False, "train_time": train_time}, file)
                time.sleep(0.05)
                model_path = r'temp_checkpoint_' + data_name + f'_{str(number + i + 1)}' + '.pth'
                model_path = os.path.join(temp_path,model_path)
                window.model_path = model_path
                continue

            save_flag = False
            try:
                with open(log_path, "r") as file:
                    log_data = json.load(file)
                    save_flag = log_data.get("save_flag", False)
            except FileNotFoundError:
                pass

            if save_flag:
                end_time = time.time()
                elapsed_time_seconds = end_time - start_time
                start_time = time.time()
                train_time = record_trian_time(train_time, elapsed_time_seconds)
                save_temp_checkpoint(data_name, epoch-1, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                                     decoder_optimizer, 0, temp_path, train_time, number + i + 1)
                time.sleep(0.05)
                with open(log_path, "w") as file:
                    json.dump({"save_flag": False, "train_time": train_time}, file)
                model_path = r'temp_checkpoint_' + data_name + f'_{str(number + i + 1)}' + '.pth'
                model_path = os.path.join(temp_path,model_path)
                window.model_path = model_path


def validate(val_loader, encoder, decoder, criterion, word_map):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    # TODO
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    losses = AverageMeter()
    top5accs = AverageMeter()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        with tqdm(total=len(val_loader), desc=f"Validate-Processing") as t:
            for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
                # Move to device, if available
                imgs = imgs.to(device)
                caps = caps.to(device)
                caplens = caplens.to(device)
                allcaps = allcaps.to(device)

                # Forward prop.
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                targets = caps_to_hot(len(scores), targets, max(decode_lengths), word_map)
                targets.to(device)

                scores_copy = scores.clone()

                # Calculate loss
                loss = criterion(scores, targets)
                # Add doubly stochastic attention regularization
                # TODO
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))

                # References
                allcaps = allcaps[sort_ind]  # because images were sorted in the decoder

                for j in range(allcaps.shape[0]):
                    img_caps = allcaps[j].tolist()
                    img_captions = list(
                        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                            img_caps))  # remove <start> and pads
                    references.append(img_captions)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)
                t.set_postfix(loss=f"{losses.val:.4f}({losses.avg:.4f})",
                              top5=f"{top5accs.val:.3f} ({top5accs.avg:.3f})")
                t.update(1)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS-{loss.avg:.3f}, TOP-5 ACCURACY-{top5.avg:.3f}, BLEU-4-{bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    def window_thread():
        global window

        word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
        word_map_file = os.path.normpath(word_map_file)

        app = QApplication(sys.argv)
        window = MainWindow(checkpoint, word_map_file)
        window.show()

        app.exec_()  # 启动 Qt 主循环


    # 创建并启动窗口线程
    window_thread = threading.Thread(target=window_thread)
    window_thread.start()

    main()
    sys.exit()
