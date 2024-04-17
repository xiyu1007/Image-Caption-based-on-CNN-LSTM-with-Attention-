import time

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import skimage
# import argparse
import cv2
from PIL import Image
import matplotlib

from utils import *

matplotlib.use('TkAgg')

from colorama import init, Fore
init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=5):
    """
    读取图像并使用 beam search 进行字幕生成。
    :param encoder: 编码器模型
    :param decoder: 解码器模型
    :param image_path: 图像路径
    :param word_map: 词映射
    :param beam_size: 每个解码步骤考虑的序列数量
    :return: 字幕，用于可视化的权重
    """

    k = beam_size
    vocab_size = len(word_map)

    # 读取图像并处理
    # 使用 cv2 读取图像
    img = cv2.imread(image_path)
    # 如果需要，将 BGR 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 检查图像是否为灰度图（2D），如果是，转换为 RGB（3D）
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 调整图像大小为 (256, 256)
    img = cv2.resize(img, (256, 256))
    # 转置维度以使通道维度成为第一个维度
    # （高度，宽度，通道）==>（通道，高度，宽度）
    img = np.transpose(img, (2, 0, 1))

    # img = img / 255.
    img = torch.FloatTensor(img / 255.)
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # 编码
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # 展平编码
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # 将问题视为批处理大小为 k
    # 将原始张量沿着第一个维度（批处理维度）复制k次，从而得到一个新的张量。
    # 新张量的尺寸是(k, num_pixels, encoder_dim)。
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # 张量以存储每一步的前 k 个前一个单词；现在它们只是 <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    """
    tensor([[504],
        [504],
        [504],
        [504],
        [504]], device='cuda:0')
    """

    # 张量以存储前 k 个序列；现在它们只是 <start>
    seqs = k_prev_words  # (k, 1)

    # 张量以存储前 k 个序列的分数；现在它们只是 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # 张量以存储前 k 个序列的 alphas；现在它们只是 1
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # 用于存储完成序列、它们的 alphas 和分数的列表
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # 开始解码
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s 是一个小于或等于 k 的数字，因为一旦序列达到 <end> 就会从这个过程中移除
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>

    score_list = [[]]
    while True:
        # k_prev_words：
        # tensor([[0.],
        #         [0.],
        #         [0.],
        #         [0.],
        #         [0.]], device='cuda:0')

        """
        nn.Embedding 层并不会自动将输入张量转换成 (batch_size, max_caption_length) 这样的维度。
        它仅仅是根据输入张量中的整数索引，从嵌入矩阵中选取对应的嵌入向量。
        在你的情况下，你传递的是一个形状为 (s, 1) 的张量给 nn.Embedding 层，其中 s 是你的 batch size 或者序列的长度。
        虽然这不是标准的形状，但是只要张量中的每个元素是有效的词汇表索引，nn.Embedding 层仍然可以处理它。
        然后，这个张量被送入嵌入层，每个整数索引将会映射到嵌入矩阵中的对应行，从而得到嵌入向量。
        因此，返回的张量形状是 (s, 1, embed_dim)，然后通过 .squeeze(1) 操作将维度为 1 的维度压缩，
        得到形状为 (s, embed_dim) 的张量。
        
        这里得到<start>对应的嵌入向量
        """
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        # # 定义解码的LSTMCell
        # self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        # decoder_dim 512

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)

        scores = F.softmax(scores, dim=1)
        # score_list[0].append(scores.tolist()[0][step - 1])

        # Add
        # scores => torch.Size([5, word_map_len])
        # top_k_scores => torch.Size([5, 1])   torch.zeros(k, 1)
        """
        对 top_k_scores 进行广播（broadcast），使其形状与 scores 相同，然后将它们逐元素相加。
        这样的操作是合理的，因为在广播时，PyTorch 会自动将形状不匹配的张量扩展为相同的形状，以便执行逐元素相加操作。
        top_k_scores.expand_as(scores) 的作用是将 top_k_scores 扩展为与 scores 相同的形状。
        在这种情况下，它将会将 top_k_scores 扩展为 (5, 506) 的形状，以便与 scores 相加。
        """
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            """
            scores[0]：选择了张量 scores 的第一行。
            假设 scores 的形状是 (batch_size, vocab_size)，那么这里选择了第一个样本的预测分数向量。
            topk(k, 0, True, True)：调用了 PyTorch 的 topk 函数，用于获取张量中最大的 k 个值及其对应的索引。
            k：表示要获取的最大值的个数。
            0：表示沿着张量的第一个维度进行操作，即对行进行操作。
            True, True：表示返回的结果按照值和索引都是按降序排列的。
            因此，top_k_scores 是一个包含了最大的 k 个值的张量，而 top_k_words 则是这些值对应的索引。
            """

            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)

        else:
            # Unroll and find top scores, and their unrolled indices

            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores

        prev_word_inds = top_k_words // vocab_size  # (s)
        # next_word_inds = top_k_words % vocab_size  # (s)

        # prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # print(Fore.BLUE + str(prev_word_inds.shape))
        # print(Fore.YELLOW + str(next_word_inds.shape))
        # print(Fore.BLUE + str(prev_word_inds))
        # print(Fore.YELLOW + str(next_word_inds))
        # print(Fore.BLUE + str(seqs[prev_word_inds].shape))
        # print(Fore.YELLOW + str(seqs[prev_word_inds]))
        # print(Fore.BLUE + str(next_word_inds.unsqueeze(1).shape))
        # print(Fore.YELLOW + str(next_word_inds.unsqueeze(1)))

        # Add new words to sequences, alphas

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1
    print(Fore.BLUE + str(complete_seqs_scores))
    time.sleep(1)
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    # print(to_caps(hot_list=score_list, is_all_hot=True))

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize((14 * 24, 14 * 24), Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    print(Fore.GREEN + "Caption=> ",end="")
    for w in words:
        print(Fore.GREEN + str(w),end=" ")

    for t in range(len(words)):
        if t > 50:
            break
        # plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)

        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap('Greys_r')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 设置参数
    model_path = 'out_data/flickr8k/save_model/checkpoint_flickr8k_5_cap_per_img_5_min_word_freq_epoch_2.pth'  # 模型路径
    img_path = f'datasets/img/COCO_test2014_000000000069.jpg'  # 图像路径
    word_map_path = 'out_data/flickr8k/out_hdf5/per_5_freq_5_maxlen_52/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  # 单词映射 JSON 路径
    model_path,_,_ = path_checker(model_path,True,False)
    img_path,_,_ = path_checker(img_path,True,False)
    word_map_path,_,_ = path_checker(word_map_path,True,False)

    beam_size = 5  # beam search 的 beam 大小
    smooth = True  # 是否进行 alpha 叠加平滑

    # 加载模型
    checkpoint = torch.load(model_path, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # 加载单词映射（word2ix）
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # 使用注意力和 beam search 进行编码和解码
    seq, alphas = caption_image_beam_search(encoder, decoder, img_path, word_map, beam_size)
    alphas = torch.FloatTensor(alphas)

    # 可视化最佳序列的标题和注意力
    visualize_att(img_path, seq, alphas, rev_word_map, smooth)
