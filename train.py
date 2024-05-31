import os.path
import sys
import threading
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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
datasets_name = 'thesis'
data_folder = f'out_data/{datasets_name}/out_hdf5/per_5_freq_1_maxlen_50'  # folder with data files saved by create_input_files.py
data_name = f'{datasets_name}_5_cap_per_img_1_min_word_freq'  # base name shared by data files
model_save_path = f'out_data/{datasets_name}/save_model'

# checkpoint = None
checkpoint = "out_data/thesis/save_model/checkpoint_thesis_5_cap_per_img_1_min_word_freq_epoch_19.pth"

is_new_epoch = True
checkpoint, _, _ = path_checker(checkpoint, True, False)
fine_tune_encoder = True  # 是否对编码器进行微调 7 begin

# Model parameters
# TODO nodel : Resnet no pre
use_pre_resnet = False
emb_dim = 1024  # 词嵌入的维度
attention_dim = 1024  # TODO 512  注意力机制中线性层的维度
decoder_dim = 1024  # 解码器RNN的维度
dropout = 0.5  # TODO 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# （仅当模型的输入具有固定大小时设置为true；否则会有很多计算开销）
cudnn.benchmark = False

# 训练参数
start_epoch = 0  # 开始的训练轮次
epochs = 20  # 训练的总轮次
epochs_since_improvement = 0  # 自上次在验证集上取得改进以来的轮次数，用于提前停止
batch_size = 64  # 32 每个训练批次中的样本数
workers = 0  # 数据加载的工作进程数 num_workers参数设置为0，这将使得数据加载在主进程中进行，而不使用多进程。
# 这个错误是由于h5py对象无法被序列化（pickled）引起的。
# 在使用多进程（multiprocessing）加载数据时，数据加载器（DataLoader）会尝试对每个批次的数据进行序列化，以便在不同的进程中传递。
encoder_lr = 1e-4  # 编码器的学习率（如果进行微调）
decoder_lr = 4e-4  # 解码器的学习率
gama_decoder = 0.98
gama_encoder = 0.95
grad_clip = 5.  # 梯度裁剪的阈值，用于防止梯度爆炸
alpha_c = 1.  # '双重随机注意力'的正则化参数
best_bleu4 = 0.  # 当前的最佳 BLEU-4 分数

train_time = "00:00:00"
start_time = 0
timeout = 3 * 60 * 60
number = 0

writer = SummaryWriter()

word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
word_map_file = os.path.normpath(word_map_file)


def main():
    """
    Training and validation.
    """
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, epochs, data_name
    global train_time, start_time, number, word_map_file, writer
    global decoder_lr, encoder_lr, gama_decoder, gama_encoder

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # 创建一个图像变换组合，包含 normalize 变换
    transform = transforms.Compose([normalize])

    # Initialize / load checkpoint
    if checkpoint is None or not os.path.exists(checkpoint):
        # LSTM
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam([{'params': decoder.parameters(), 'initial_lr': decoder_lr}],
                                             lr=decoder_lr)
        # ResNet
        encoder = Encoder(use_pre_resnet)
        encoder.fine_tune(fine_tune_encoder)
        if fine_tune_encoder:
            encoder_optimizer = torch.optim.Adam(
                [{'params': [p for p in encoder.resnet.parameters() if p.requires_grad], 'initial_lr': encoder_lr}],
                lr=encoder_lr)
        else:
            encoder_optimizer = None
    else:
        checkpoint = torch.load(checkpoint)
        train_time = checkpoint['train_time']
        if is_new_epoch:
            start_epoch = checkpoint['epoch'] + 1
            number = 0
        else:
            start_epoch = checkpoint['epoch']
            number = checkpoint['number'] + 1

        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                [{'params': [p for p in encoder.resnet.parameters() if p.requires_grad], 'initial_lr': encoder_lr}],
                lr=encoder_lr)

        window.set_train_time(train_time, number - 1, start_epoch, 0)

    # print thesis info
    if checkpoint is not None:
        print_model_info(checkpoint)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
    # Epochs
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        subset_indices = list(range(number * batch_size, len(dataset)))
        # 使用 Subset 类来创建数据集的部分子集
        subset_dataset = Subset(dataset, subset_indices)
        train_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
        total_batch = (len(dataset) // batch_size) + 1
        print("=" * 100)
        print(Fore.BLUE + "Device：", Fore.BLUE + str(device))
        print(Fore.BLUE + f'epoch: {epoch}, batch: {number}, total_batch: {total_batch}, remain: {len(train_loader)}')
        decoder_optimizer.param_groups[0]['initial_lr'] = decoder_optimizer.param_groups[0]['lr']
        print(Fore.BLUE + "initial_lr_decoder：", str(decoder_optimizer.param_groups[0]['lr']))
        decoder_scheduler = CosineAnnealingLR(optimizer=decoder_optimizer,
                                              T_max=total_batch, eta_min=decoder_lr * (gama_decoder ** (epoch + 1)),
                                              last_epoch=number)
        if fine_tune_encoder and encoder_optimizer is not None:
            print(Fore.BLUE + "Allow fine_tune_encoder")
            encoder_optimizer.param_groups[0]['initial_lr'] = encoder_optimizer.param_groups[0]['lr']
            print(Fore.BLUE + "initial_lr_encoder：", str(encoder_optimizer.param_groups[0]['lr']))
            encoder_scheduler = CosineAnnealingLR(optimizer=encoder_optimizer,
                                                  T_max=total_batch, eta_min=encoder_lr * (gama_encoder ** (epoch + 1)),
                                                  last_epoch=number)
        else:
            encoder_scheduler = None
            print(Fore.BLUE + "Don`t fine_tune_encoder")

        # 如果连续 20 个 epoch 都没有性能提升，则提前终止训练
        if epochs_since_improvement == 10:
            print("Reaching epochs_since_improvement")
            break
        # 假设训练开始
        # # One epoch's training
        log_path = fr'.\logs\{datasets_name}\train\epoch_{epoch}'
        temp_log_path = fr'.\logs\{datasets_name}\train\temp_epoch_{epoch}'
        log_write(src_dir=log_path, dst_dir=temp_log_path)
        time.sleep(0.05)
        writer = SummaryWriter(temp_log_path)
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              encoder_scheduler=encoder_scheduler,
              decoder_scheduler=decoder_scheduler,
              epoch=epoch,
              word_map=word_map,
              total_batch=total_batch,
              temp_log_path=temp_log_path,
              log_path=log_path)

        # One epoch's validation
        validate_path = fr'.\logs\{datasets_name}\validate\epoch_{epoch}'
        log_write(src_dir=validate_path, dst_dir=validate_path)
        writer = SummaryWriter(validate_path)
        time.sleep(0.01)
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
        number = 0
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best,
                        model_save_path, train_time, None, None, number=number)
        time.sleep(0.01)
        log_write(temp_log_path, log_path)
        time.sleep(1)
    writer.close()


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer,
          encoder_scheduler, decoder_scheduler,
          epoch, word_map, total_batch, temp_log_path, log_path):
    """
    Performs one epoch's training.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        encoder (nn.Module): Encoder thesis.
        decoder (nn.Module): Decoder thesis.
        criterion (nn.Module): Loss layer.
        encoder_optimizer (torch.optim.Optimizer): Optimizer to update encoder's weights (if fine-tuning).
        decoder_optimizer (torch.optim.Optimizer): Optimizer to update decoder's weights.
        encoder_scheduler: Scheduler for adjusting learning rate of encoder.
        decoder_scheduler: Scheduler for adjusting learning rate of decoder.
        epoch (int): Epoch number.
        word_map (dict): Word map.
        total_batch (int): Total number of batches.
        temp_log_path (str): Path to temporary log file.
        log_path (str): Path to log file.
    """
    global train_time, start_time, number, window, timeout
    window.enable_button()  # 设置按钮为可点击状态

    # 设置解码、编码器为训练模式（启用 dropout 和批归一化）
    decoder.train()
    encoder.train()

    # # 用于记录每个单词的损失的指标
    losses = AverageMeter()  # loss (per word decoded)
    # 用于记录 top-5 准确率的指标
    top5accs = AverageMeter()  # top5 accuracy

    end_time = start_time

    # Batches
    with tqdm(total=total_batch, initial=number, desc=f"Training: Epoch {epoch}/{epochs}") as t:
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            # print(len(train_loader))
            # print(i)
            # print(number)
            # print(data_len)
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
            # targets = targets.to(device)

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
            lr_decoder = decoder_optimizer.param_groups[0]['lr']

            if encoder_optimizer is not None:
                encoder_optimizer.step()
                lr_encoder = encoder_optimizer.param_groups[0]['lr']
            else:
                lr_encoder = 0

            decoder_scheduler.step()
            if encoder_optimizer is not None:
                encoder_scheduler.step()

            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))

            writer.add_scalars('Train/learning_rate',
                               {'decoder_optimizer': lr_decoder, 'encoder_optimizer': lr_encoder}, number + i)
            writer.add_scalars('Train/Loss', {'val': losses.val, 'avg': losses.avg}, number + i)
            writer.add_scalars('Train/Top5Accuracy', {'val': top5accs.val, 'avg': top5accs.avg}, number + i)

            elapsed_time_seconds = time.time() - start_time
            try:
                if not window.set_train_time(record_trian_time(train_time, elapsed_time_seconds), number + i, epoch,
                                             lr_decoder):
                    sys.exit()  # 使用 os._exit() 来直接退出整个程序
            except Exception as e:
                print(Fore.YELLOW + "\nError train set_train_time: ", e)

            if window.lr_flag:
                window.lr_flag = False
                adjust_learning_rate(decoder_optimizer, float(window.spin_lr.text()))
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, float(window.spin_lr.text()))

            if window.save_flag or time.time() - end_time > timeout:
                if not window.save_flag:
                    end_time = time.time()
                start_time = time.time()
                train_time = record_trian_time(train_time, elapsed_time_seconds)
                save_temp_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,
                                     encoder_optimizer,
                                     decoder_optimizer, 0, model_save_path, train_time, losses, top5accs,
                                     number + i)

                time.sleep(0.05)
                log_write(temp_log_path, log_path)
                time.sleep(0.05)
                model_path = 'temp_checkpoint_' + data_name + f'_epoch_{epoch}_batch_{str(number + i)}' + '.pth'
                model_path = os.path.join(model_save_path, model_path)
                window.model_path = model_path
                if window.save_flag:
                    window.save_flag = False
                    is_continue = window.get_continue_flag()
                    window.save_recall(window.msg_box_saving)
                    if not is_continue:
                        time.sleep(3)
                        print(Fore.YELLOW + "\nTraining halted by user request.")
                        window.ban_button()
                        window.main_flag = False
                        window.button_predict.setText("预测")  # 设置按钮为不可点击状态
                        window.enable_button()  # 设置按钮为不可点击状态
                        # os._exit()
                        sys.exit()  # 使用 os._exit() 来直接退出整个程序
            t.set_postfix(loss=f"{losses.val:.4f}({losses.avg:.4f})",
                          top5=f"{top5accs.val:.3f}% ({top5accs.avg:.3f}%)")
            t.update(1)
        window.ban_button()


def validate(val_loader, encoder, decoder, criterion, word_map, write_log=True):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder thesis
    :param decoder: decoder thesis
    :param criterion: loss layer
    :param word_map: Word map.
    :return: BLEU-4 score
    """
    global writer
    # TODO
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    losses = AverageMeter()
    top5accs = AverageMeter()

    bleu = AverageMeter()
    bleu1 = AverageMeter()
    bleu2 = AverageMeter()
    bleu3 = AverageMeter()
    bleu4 = AverageMeter()
    rouge = AverageMeter()
    rouge1 = AverageMeter()
    rouge2 = AverageMeter()
    rouge3 = AverageMeter()
    rouge4 = AverageMeter()

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
                targets = targets.to(device)

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
                    temp_preds.append(preds[j][:decode_lengths[j]])
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)
                # Calculate BLEU-4 scores
                ([b1, b2, b3, b4], b_avg) = get_bleu(references, hypotheses, all_bleu=True)
                r_scores_corpus, r_avg = get_rouge(references, hypotheses, all_rouge=True)
                r1 = r_scores_corpus[0]
                r2 = r_scores_corpus[1]
                r3 = r_scores_corpus[2]
                r4 = r_scores_corpus[3]
                references = []
                hypotheses = []
                bleu.update(b_avg)
                bleu1.update(b1)
                bleu2.update(b2)
                bleu3.update(b3)
                bleu4.update(b4)
                rouge.update(r_avg)
                rouge1.update(r1['f'])
                rouge2.update(r2['f'])
                rouge3.update(r3['f'])
                rouge4.update(r4['f'])
                # Write validation metrics to TensorBoard
                if write_log:
                    writer.add_scalars('Test/bleu', {'bleu1': b1, 'bleu1_avg': bleu1.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu2': b2, 'bleu2_avg': bleu2.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu3': b3, 'bleu3_avg': bleu3.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu4': b4, 'bleu4_avg': bleu4.avg}, i)
                    writer.add_scalars('Test/bleu', {'bleu_avg(1-4)': bleu.avg}, i)

                    writer.add_scalars('Test/rouge',
                                       {'r1_F1': r1['f'], 'r1_precision': r1['p'],
                                        'r1_recall': r1['r'], 'r1_F1_avg': rouge1.avg}, i)
                    writer.add_scalars('Test/rouge',
                                       {'r2_F1': r2['f'], 'r2_precision': r2['p'],
                                        'r2_recall': r2['r'], 'r2_F1_avg': rouge2.avg}, i)
                    writer.add_scalars('Test/rouge',
                                       {'r3_F1': r3['f'], 'r3_precision': r3['p'],
                                        'r3_recall': r3['r'], 'r3_F1_avg': rouge3.avg}, i)
                    writer.add_scalars('Test/rouge',
                                       {'r4_F1': r4['f'], 'r4_precision': r4['p'],
                                        'r4_recall': r4['r'], 'r4_F1_avg': rouge4.avg}, i)
                    writer.add_scalars('Test/rouge', {'rouge_avg(1-4)': rouge.avg}, i)

                # Write validation metrics to TensorBoard
                writer.add_scalars('Train/Loss', {'val': losses.val, 'avg': losses.avg}, i)
                writer.add_scalars('Train/Top5Accuracy', {'val': top5accs.val, 'avg': top5accs.avg}, i)
                writer.add_scalars('Validation/bleu4', {'val': bleu.val, 'avg': bleu.avg}, i)
                writer.add_scalars('Validation/rouge', {'val': rouge.val, 'avg': rouge.avg}, i)

                t.set_postfix(loss=f"{losses.val:.4f}({losses.avg:.4f})",
                              top5=f"{top5accs.val:.3f} ({top5accs.avg:.3f})")
                t.update(1)
        print(
            '\n * LOSS: {loss.avg:.3f}, TOP-5: {top5.avg:.3f}\n'
            'BLEU: {bleu.avg}, Rouge: {rouge.avg}'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu,
                rouge=rouge))

    return (rouge.avg + bleu.avg) / 2.0


if __name__ == '__main__':
    def window_thread():
        global window, checkpoint, word_map_file
        try:
            app = QApplication(sys.argv)  # 创建 QApplication 实例
            # 设置应用程序的窗口图标
            app.setWindowIcon(QIcon('utils/main.jpg'))

            # 创建 QFont 实例并设置字体和字号
            font = QFont()
            font.setFamily("Arial")  # 设置字体
            font.setPointSize(12)  # 设置字号

            # 将 QFont 应用到应用程序上的所有 QLineEdit 控件
            app.setFont(font, "QLineEdit")
            app.setFont(font, "QPushButton")
            app.setFont(font, "QMessageBox")
            app.setFont(font, "QDoubleSpinBox")
            app.setFont(font, "QLabel")

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
    # window_thread.join()
    sys.exit()
