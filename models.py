import torch
from torch import nn
import torchvision
from torchsummary import summary
from colorama import init, Fore

# 初始化 colorama
init()

from utils import path_checker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置下载路径
models_download_path = 'models/ResNet'
models_download_path, _, _ = path_checker(models_download_path)
torch.hub.set_dir(models_download_path)


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        # 设置编码后的图像大小
        self.encoded_image_size = encoded_image_size
        resnet = torchvision.models.resnet101 \
            (weights=torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V2)
        # 移除线性层和池化层（因为我们不进行分类任务）
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # 调整特征图大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # 执行微调
        self.fine_tune()

    def forward(self, images):
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=False):
        """
        允许或禁止对编码器的卷积块2到4进行梯度计算。
        :param fine_tune: 是否允许微调?
        """
        # 禁止/允许对整个 ResNet 的梯度计算
        # TODO
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune

        # 如果允许微调，仅允许微调卷积块2到4
        # if fine_tune:
        #     for c in list(self.resnet.children())[4:]:
        #         for p in c.parameters():
        #             p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim=2048, decoder_dim=512, attention_dim=512):
        super(Attention, self).__init__()
        # 通过线性层将编码后的图像特征映射到注意力网络的维度
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # 通过线性层将解码器的输出映射到注意力网络的维度
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # 通过线性层计算进行 softmax 的权重值
        self.full_att = nn.Linear(attention_dim, 1)
        # 激活函数 ReLU
        self.relu = nn.ReLU()
        # softmax 层，用于计算权重值
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        前向传播。
        :param encoder_out: 编码后的图像特征，维度为 (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: 上一个解码器输出，维度为 (batch_size, decoder_dim)
        :return: 注意力加权编码，权重
        """
        # 使用线性层将编码后的图像特征映射到注意力网络的维度
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        # 使用线性层将解码器的输出映射到注意力网络的维度
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # 计算注意力权重
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)

        # 使用 softmax 计算权重值
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        # 计算注意力加权编码
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.2):
        """
        :param attention_dim: attention网络的大小
        :param embed_dim: 嵌入层的维度大小
        :param decoder_dim: 解码器的RNN维度大小
        :param vocab_size: 词汇表的大小
        :param encoder_dim: 编码图像的通道大小，默认为2048
        :param dropout: dropout的比例
        """
        super(DecoderWithAttention, self).__init__()

        # 保存参数
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # 定义注意力网络
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout, inplace=True)

        # 定义解码的LSTMCell
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # 定义线性层以找到LSTMCell的初始隐藏状态
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

        # 定义线性层以找到LSTMCell的初始细胞状态
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # 定义线性层以创建一个sigmoid激活的门
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义线性层以在词汇表上找到分数
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # 初始化一些层的权重
        self.init_weights()

    def init_weights(self):
        """
        用均匀分布的值初始化一些参数，以便更容易地进行收敛。
        """
        # 初始化嵌入层的权重，使用均匀分布在(-0.1, 0.1)之间
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # 初始化线性层的偏置，将所有元素设置为0
        self.fc.bias.data.fill_(0)
        # 初始化线性层的权重，使用均匀分布在(-0.1, 0.1)之间
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        使用预训练的嵌入加载嵌入层。
        :param embeddings: 预训练的嵌入
        """
        # 将嵌入层的权重设置为预训练的嵌入
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        允许微调嵌入层吗？（如果使用预训练的嵌入，不允许微调是有意义的。）
        :param fine_tune: 是否允许微调
        """
        # 设置嵌入层的requires_grad属性，以决定是否允许微调
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        根据编码的图像创建解码器的LSTM的初始隐藏状态和细胞状态。
        :param encoder_out: 编码的图像，维度为 (batch_size, num_pixels, encoder_dim)
        :return: 隐藏状态，细胞状态
        """
        # 对编码的图像进行平均，得到 (batch_size, encoder_dim) 的张量
        mean_encoder_out = encoder_out.mean(dim=1)
        # mean_encoder_out = encoder_out.max(dim=1)

        # 使用线性层找到LSTM的初始隐藏状态
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        # 使用线性层找到LSTM的初始细胞状态
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        # 获取维度信息
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # 将图像展平
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)  # encoded_image_size*encoded_image_size = 14 * 14

        # 按长度降序排列输入数据
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # sort_id 排序前的索引

        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # 嵌入层
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # 初始化LSTM状态
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # 由于在生成<end>后我们就完成了生成，所以我们不会在<end>位置解码
        # 因此，解码长度实际上是长度 - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # 创建张量以保存词预测分数和注意力权重
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # 在每个时间步进行解码
        # 通过基于解码器先前隐藏状态输出的注意力权重来加权编码器的输出
        # 然后使用先前的单词和注意力加权编码生成解码器中的新单词

        for t in range(max(decode_lengths)):
            # 计算当前时间步的批次大小
            batch_size_t = sum([l > t for l in decode_lengths])
            # decode_lengths已经降序
            # 使用注意力模型计算注意力加权的编码器输出和注意力权重
            # h,encoder_out => batch_size, encoded_image_size * encoded_image_size, 2048
            # encoder_out[:batch_size_t] => 0:batch_size_t, encoded_image_size * encoded_image_size, 2048
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # 计算门控标量，用于调整注意力加权的编码器输出
            """
            self.f_beta()：这是一个神经网络层，它将隐藏状态 h[:batch_size_t] 映射到与编码器输出的维度相同的空间（encoder_dim）。
            这个映射是为了与编码器输出进行加权求和。
            """
            # 定义线性层以创建一个sigmoid激活的门
            # self.f_beta = nn.Linear(decoder_dim, encoder_dim)
            # 解码器的隐藏状态被调整为与编码器输出相同的维度
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            """
            embeddings => [batch_size, max_caption_length, embedding_dim]
            attention_weighted_encoding => (batch_size, encoder_dim)
            """

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            """
            将当前时间步的嵌入（embeddings）和注意力加权的编码（attention_weighted_encoding）进行拼接。
            通常，解码器在每个时间步会使用当前时间步的嵌入以及注意力机制得到的编码信息来生成下一个单词。
            """

            # 通过全连接层生成词预测分数
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            # 将预测分数存储到predictions张量中

            predictions[:batch_size_t, t, :] = preds
            # 将注意力权重存储到alphas张量中
            alphas[:batch_size_t, t, :] = alpha
        # 返回模型的输出：词预测分数、排序后的编码字幕、解码长度、注意力权重、排序索引
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind


if __name__ == '__main__':
    # 创建模型实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    # 将模型移动到可用的设备上
    encoder.to(device)
    # 打印模型摘要
    summary(encoder, input_size=(3, 224, 224))
    # AdaptiveAvgPool2d - 343[-1, 2048, 14, 14]