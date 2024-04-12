import torch
from torch import nn
import torchvision
from torchsummary import summary

from utils import path_checker

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

    def fine_tune(self, fine_tune=True):
        """
        允许或禁止对编码器的卷积块2到4进行梯度计算。
        :param fine_tune: 是否允许微调?
        """
        # 禁止对整个 ResNet 的梯度计算
        for p in self.resnet.parameters():
            p.requires_grad = False

        # 如果允许微调，仅允许微调卷积块2到4
        if fine_tune:
            for c in list(self.resnet.children())[4:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


if __name__ == '__main__':
    # 创建模型实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()
    # 将模型移动到可用的设备上
    encoder.to(device)
    # 打印模型摘要
    summary(encoder, input_size=(3, 224, 224))
    # AdaptiveAvgPool2d - 343[-1, 2048, 14, 14]
