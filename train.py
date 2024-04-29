import os.path
import sys
import threading
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from torch import nn
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from Win_Qt import MainWindow
from datasets import *
from models import Encoder, DecoderWithAttention
from utils import *
from utils_eval import *

init()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# TODO 修改路径
datasets_name = 'flickr8k'
writer = SummaryWriter()
data_folder = f'out_data/{datasets_name}/out_hdf5/per_5_freq_1_maxlen_50'  # folder with data files saved by create_input_files.py
data_name = f'{datasets_name}_5_cap_per_img_1_min_word_freq'  # base name shared by data files
model_save_path = f'out_data/{datasets_name}/save_model'

checkpoint = None
is_new_epoch = False
checkpoint, _, _ = path_checker(checkpoint, True, False)

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
decoder_lr = 4e-4  # 解码器的学习率
grad_clip = 5.  # 梯度裁剪的阈值，用于防止梯度爆炸
alpha_c = 1.  # '双重随机注意力'的正则化参数
best_bleu4 = 0.  # 当前的最佳 BLEU-4 分数
print_freq = 100  # 每训练多少个批次打印一次训练/验证统计信息
fine_tune_encoder = True  # 是否对编码器进行微调

train_time = "00:00:00"
start_time = 0
timeout = 2 * 60 * 60
number = 0

word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
word_map_file = os.path.normpath(word_map_file)


# checkpoint = None

def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name
    global word_map, train_time, start_time, number, word_map_file, writer

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
    else:
        checkpoint = torch.load(checkpoint)
        train_time = checkpoint['train_time']
        number = checkpoint['number'] + 1
        if is_new_epoch:
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = checkpoint['epoch']
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
        window.set_train_time(train_time, number - 1, start_epoch)
        print(Fore.GREEN + 'Model loading from => \n' + str(checkpoint))
        time.sleep(0.2)
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
    # dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
    #     sampler=torch.utils.data.sampler.SequentialSampler(range(number * batch_size, len(dataset))))
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    # TODO
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        torch.manual_seed(52 + (epoch * 100))
        dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform)
        subset_indices = list(range(number * batch_size, len(dataset)))
        # 使用 Subset 类来创建数据集的部分子集
        subset_dataset = Subset(dataset, subset_indices)
        train_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
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
        # # One epoch's training
        writer = SummaryWriter(fr'.\logs\{datasets_name}\train\{datasets_name}_{epoch}')
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              word_map=word_map,
              data_len=len(dataset) // batch_size)

        # One epoch's validation
        writer = SummaryWriter(fr'.\logs\{datasets_name}\validate\{datasets_name}_{epoch}')
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
        elapsed_time_seconds = time.time() - start_time
        start_time = time.time()
        train_time = record_trian_time(train_time, elapsed_time_seconds)
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, model_save_path, train_time, number=0)
        number = 0

        time.sleep(1)

    writer.close()


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, word_map, data_len):
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

    global train_time, start_time, number, window, timeout

    # 设置解码、编码器为训练模式（启用 dropout 和批归一化）
    decoder.train()
    encoder.train()

    # 用于记录每个单词的损失的指标
    losses = AverageMeter()  # loss (per word decoded)
    # 用于记录 top-5 准确率的指标
    top5accs = AverageMeter()  # top5 accuracy
    BEST_acc = 0
    batch_since_improvement = 0
    min_improvement = 50
    end_time = start_time

    # Batches
    with tqdm(total=data_len, initial=number, desc=f"Training: Epoch {epoch}/{epochs}") as t:
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            # print(to_caps(caps, False, word_map))
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
            #  TODO 5831
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

            if top5accs.val > BEST_acc:
                BEST_acc = top5accs.val
                batch_since_improvement = 0
            else:
                batch_since_improvement += 1
            if batch_since_improvement > min_improvement:
                # 调整解码器（decoder）的学习率，将当前学习率乘以 0.8
                adjust_learning_rate(decoder_optimizer, 0.85)
                # 如果需要对编码器（encoder）进行微调，也调整编码器的学习率
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.85)

            writer.add_scalar('Train/learning_rate',decoder_optimizer.param_groups[0]['lr'], i)
            # 在损失和准确率更新后
            writer.add_scalars('Train/Loss', {'val': losses.val, 'avg': losses.avg}, i)
            writer.add_scalars('Train/Top5Accuracy', {'val': top5accs.val, 'avg': top5accs.avg}, i)

            # if i % 5 == 0:
            elapsed_time_seconds = time.time() - start_time
            window.set_train_time(record_trian_time(train_time, elapsed_time_seconds), number + i, epoch)

            if window.save_flag or time.time() - end_time > timeout:
                if not window.save_flag:
                    end_time = time.time()
                start_time = time.time()
                train_time = record_trian_time(train_time, elapsed_time_seconds)
                save_temp_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,
                                     encoder_optimizer,
                                     decoder_optimizer, 0, model_save_path, train_time, number + i)

                time.sleep(0.05)
                model_path = 'temp_checkpoint_' + data_name + f'_epoch_{epoch}_batch_{str(number)}' + '.pth'
                model_path = os.path.join(model_save_path, model_path)
                window.model_path = model_path
                # window.set_train_time(train_time, number + i, epoch)
                if window.save_flag:
                    window.save_recall(window.msg_box_saving)
                    window.save_flag = False

            t.set_postfix(loss=f"{losses.val:.4f}({losses.avg:.4f})",
                          top5=f"{top5accs.val:.3f}% ({top5accs.avg:.3f}%)")
            t.update(1)


def validate(val_loader, encoder, decoder, criterion, word_map):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    global writer
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
                              top5=f"{losses.val:.3f} ({top5accs.avg:.3f})")
                t.update(1)
                # Calculate BLEU-4 scores
                bleu = get_bleu(references, hypotheses)
                rouge = get_rouge(references, hypotheses)

                # Write validation metrics to TensorBoard
                writer.add_scalars('Train/Loss', {'val': losses.val, 'avg': losses.avg}, i)
                writer.add_scalars('Train/Top5Accuracy', {'val': top5accs.val, 'avg': top5accs.avg}, i)
                writer.add_scalar('Validation/bleu4', bleu, i)
                writer.add_scalar('Validation/rouge', rouge, i)

        print(
            '\n * LOSS-{loss.avg:.3f}, TOP-5-{top5.avg:.3f}, BLEU_4-{bleu}, Rouge-{rouge}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu,
                rouge=rouge))

    return (rouge + bleu)/2.0


if __name__ == '__main__':
    def window_thread():
        global window, checkpoint, word_map_file
        try:
            app = QApplication(sys.argv)  # 创建 QApplication 实例
            # 创建 QFont 实例并设置字体和字号
            font = QFont()
            font.setFamily("Arial")  # 设置字体
            font.setPointSize(12)  # 设置字号
            # 将 QFont 应用到应用程序上的所有 QLineEdit 控件
            app.setFont(font, "QLineEdit")
            app.setFont(font, "QPushButton")

            window = MainWindow(checkpoint, word_map_file)
            window.show()
            # app.exec_()  # 返回主循环的退出代码
            sys.exit(app.exec_())  # 返回主循环的退出代码
        except Exception as e:
            print(Fore.YELLOW + "\n" + str(e))


    # 创建并启动窗口线程

    window_thread = threading.Thread(target=window_thread)
    window_thread.start()

    main()
    # 等待窗口线程执行完成后再退出
    window_thread.join()
    sys.exit()
