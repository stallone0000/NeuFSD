import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet
try:
    from torchsummary import summary
except ImportError:
    summary = None
import timm
from timm.models.vision_transformer import VisionTransformer

class Two_dim_CNN(nn.Module):
    def __init__(self, in_channel, out_shape, conv1_dim, conv2_dim):
        super(Two_dim_CNN, self).__init__()
        self.conv0 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, conv1_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(conv1_dim),
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, conv1_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(conv1_dim),
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, conv1_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(conv1_dim),
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, conv1_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(conv1_dim),
            nn.Conv2d(conv1_dim, conv2_dim, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.get_weight = nn.AdaptiveAvgPool2d((1,1))
        self.Linear = nn.Sequential(
            nn.Linear(conv2_dim*4, conv2_dim*4),
            nn.ReLU(),
            nn.Linear(conv2_dim*4, out_shape),
        )
    def conv(self,x_ls,conv_ls):
        x_out = [conv_ls[index](x) for index,x in enumerate(x_ls)]
        return x_out
    # 方法1：广播乘法 + 空间求和
    def channel_wise_sum_v1(self, weights, x):
        weighted_x = weights * x  # 广播乘法 (32,1024,32,32)
        return weighted_x.sum(dim=(2,3), keepdim=False)  # 空间求和

    def forward(self, x):
        x_0,x_1,x_2,x_3 = self.conv([x,x,x,x],[self.conv0,self.conv1,self.conv2,self.conv3])
        x = torch.cat((x_0,x_1,x_2,x_3), dim=1)
        weights = self.get_weight(x)
        weights = F.softmax(weights,dim=1)
        weighted_x = self.channel_wise_sum_v1(weights, x)
        output = self.Linear(weighted_x)
        return output

class r34(nn.Module):
    def __init__(self, out_dim):
        super(r34, self).__init__()
        self.resnet = resnet.resnet34()
        num_features = self.resnet.fc.in_features
        # 替换原始全连接层，直接输出到out_dim
        self.resnet.fc = nn.Linear(num_features, out_dim)
    
    def forward(self, x):
        x = self.resnet(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2**14),
            nn.ReLU(),
            nn.Linear(2**14, 2**13),
            nn.ReLU(),
            nn.Linear(2**13, 2**12),
            nn.ReLU(),
            nn.Linear(2**12, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
    
class CustomViT(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        # 自定义ViT参数
        self.vit = VisionTransformer(
            img_size=64,        # 输入尺寸
            patch_size=16,       # 分块大小 (128/16=8 -> 8x8 patches)
            in_chans=3,          # 输入通道数
            embed_dim=768,       # 嵌入维度
            depth=12,            # Transformer层数
            num_heads=12,        # 注意力头数
            mlp_ratio=4.0,       # MLP扩展比例
            qkv_bias=True,       # 是否使用QKV偏置
            num_classes=0        # 禁用分类头
        )
        
        # 自定义回归头
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1000),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1000, out_dim)
        )

    def forward(self, x):
        # 提取特征 [batch_size, num_patches+1, embed_dim]
        x = self.vit(x)  
        # # 取分类token [batch_size, embed_dim]
        # cls_token = x[:, 0] 
        return self.head(x)
    
if __name__ == '__main__':
    # model = Two_dim_CNN(in_channel=1, out_shape=20, conv1_dim=128, conv2_dim=512)
    # 使用示例
    model = CustomViT(out_dim=20).to('cuda')
    model = r34(out_dim=20).to('cuda')
    # 测试前向传播
    # test_input = torch.randn(32, 3, 128, 128).to('cuda')
    # print(model(test_input).shape)
    if summary is not None:
        summary(model, (3, 128, 128))  # 输入形状验证
