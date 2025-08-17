import torch 
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """空间注意力机制"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, 1, kernel_size=1)
        
    def forward(self, x):
        # 通道压缩
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        
        # 生成注意力权重
        attn = torch.sigmoid(attn)
        
        # 应用注意力
        return x * attn.expand_as(x)

class ResidualBlock(nn.Module):
    """扩展卷积残差块"""
    def __init__(self, in_channels, out_channels, dilation=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=dilation, dilation=dilation, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 处理输入通道数与输出通道数不匹配或步长不为1的情况
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity
        out = self.relu(out)
        return out

def visualize_tensor(tensor, title="Tensor Visualization"):
    """使用matplotlib渲染张量的每个通道"""
    import matplotlib.pyplot as plt
    import numpy as np

    tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 4:  # Batch size, Channels, Height, Width
        tensor = tensor[0]  # 取第一个样本
    elif tensor.ndim == 3:  # Channels, Height, Width
        pass  # 保持不变
    else:
        raise ValueError("Unsupported tensor shape for visualization")

    # 转换为适合渲染的格式
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 [0, 1]

    num_channels = tensor.shape[0]
    if num_channels == 1:
        plt.imshow(tensor[0], cmap="gray")
    else:
        fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
        fig.suptitle(title)

        for i in range(num_channels):
            axes[i].imshow(tensor[i], cmap="viridis")
            axes[i].set_title(f"Channel {i + 1}")
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

class ImprovedEncoder(nn.Module):
    """改进的编码器（输出扁平化）"""
    def __init__(self, input_channels=3):
        super(ImprovedEncoder, self).__init__()
        
        # 初始卷积，保持空间尺寸
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # 第一个下采样，减小空间尺寸 15x30 -> 8x15
        self.down1 = ResidualBlock(32, 32, stride=2)
        
        # 第一个残差块组
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32, dilation=1),
            ResidualBlock(32, 64, dilation=2)
        )
        
        # 第二个下采样，进一步减小空间尺寸 8x15 -> 4x8
        self.down2 = ResidualBlock(64, 64, stride=2)
        
        # 第二个残差块组
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 64, dilation=1),
            ResidualBlock(64, 128, dilation=2)
        )
        
        # 添加最大池化层，进一步减小尺寸 4x8 -> 2x4
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 注意力块
        self.attn = AttentionBlock(128)
        
        # 记录特征维度，用于解码器重构
        self.feature_size = (128, 2, 4)
        
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 第一个下采样
        x = self.down1(x)
        
        # 第一个残差块组
        x = self.layer1(x)
        
        # 第二个下采样
        x = self.down2(x)
        
        # 第二个残差块组
        x = self.layer2(x)
        
        # 池化层
        x = self.pool(x)
        
        # 注意力机制
        x = self.attn(x)
        
        # 扁平化
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return x


class ImprovedDecoder(nn.Module):
    """改进的解码器（适配扁平化的编码器输出）"""
    def __init__(self, input_dim=128*2*4, output_channels=1):
        super(ImprovedDecoder, self).__init__()
        
        self.feature_size = (128, 2, 4)
        
        # 从扁平特征重构为3D特征图
        self.fc = nn.Linear(input_dim, self.feature_size[0] * self.feature_size[1] * self.feature_size[2])
        
        # 第一个上采样 2x4 -> 4x8
        self.up1 = nn.ConvTranspose2d(self.feature_size[0], 64, kernel_size=4, stride=2, padding=1)
        self.bn_up1 = nn.BatchNorm2d(64)
        
        # 第二个上采样 4x8 -> 8x16
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn_up2 = nn.BatchNorm2d(32)
        
        # 第三个上采样 8x16 -> 16x32
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn_up3 = nn.BatchNorm2d(16)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 输出层
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)
        
        # 精确调整大小的层
        self.size_adjust = nn.Upsample(size=(15, 30), mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # 从扁平特征重构为3D特征图
        x = self.fc(x)
        x = x.view(-1, self.feature_size[0], self.feature_size[1], self.feature_size[2])

        
        # 第一个上采样
        x = self.up1(x)
        x = self.bn_up1(x)
        x = self.relu(x)
        
        # 第二个上采样
        x = self.up2(x)
        x = self.bn_up2(x)
        x = self.relu(x)
        
        # 第三个上采样
        x = self.up3(x)
        x = self.bn_up3(x)
        x = self.relu(x)
        
        # 输出层
        x = self.final_conv(x)
        
        # 确保尺寸正确
        x = self.size_adjust(x)

        return torch.sigmoid(x)  # 确保输出在 [0, 1] 范围内

class OneShotPathPlanner(nn.Module):
    """改进的端到端路径规划模型（扁平化编码器输出）"""
    def __init__(self, input_channels=3, output_channels=1):
        super(OneShotPathPlanner, self).__init__()
        self.encoder = ImprovedEncoder(input_channels)
        # 计算编码器输出的扁平维度
        flat_size = 128 * 2 * 4  # 通道 * 高度 * 宽度
        self.decoder = ImprovedDecoder(flat_size, output_channels)
        
    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output

    
if __name__ == "__main__":
    # 测试模型
    model = OneShotPathPlanner(input_channels=3, output_channels=1)
    sample_input = torch.randn(1, 3, 15, 30)  # Batch size 1, 3 channels, 15x30 grid
    output = model(sample_input)
    print("Output shape:", output.shape)  # 应该是 (1, 1, 15, 30)