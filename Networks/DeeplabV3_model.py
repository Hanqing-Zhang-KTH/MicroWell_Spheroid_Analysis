import os
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
from torchvision.models._utils import IntermediateLayerGetter
from functools import partial

# --------------------------------------------------
# Helper: Extract Accuracy from Filename
# --------------------------------------------------

def get_accuracy_from_filename(filename):
    basename = os.path.basename(filename)
    # Look for best_model_acc_X.XXX_loss_Y.YYYY.pth pattern
    match = re.search(r'best_model_acc_(\d+(?:\.\d+)?)_loss_\d+(?:\.\d+)?\.pth$', basename)
    if match:
        return float(match.group(1))
    # Fallback to LastNet pattern for backward compatibility
    match = re.search(r'LastNet_(\d+(?:\.\d+)?).pth$', basename)
    if match:
        return float(match.group(1))
    return None

def get_best_model_path(weight_dir):
    # First try to find best_model files
    weight_files = glob.glob(os.path.join(weight_dir, 'best_model_*.pth'))
    if not weight_files:
        # Fallback to LastNet files for backward compatibility
        weight_files = glob.glob(os.path.join(weight_dir, 'LastNet_*.pth'))
        if not weight_files:
            raise FileNotFoundError(f"No 'best_model_*.pth' or 'LastNet_*.pth' files in {weight_dir}")
    best_file = max(weight_files, key=lambda x: get_accuracy_from_filename(x) or -1)
    return best_file

# --------------------------------------------------
# DeepLabV3+ with Elastic Weight Consolidation
# --------------------------------------------------

class ElasticWeightConsolidation:
    def __init__(self, model, fisher_diagonal, optpar):
        self.model = model
        self.fisher_diagonal = fisher_diagonal
        self.optpar = optpar
        self.lambda_ewc = 0.4  # EWC超参数

    def loss(self):
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_diagonal:
                loss += torch.sum(self.fisher_diagonal[name] * (param - self.optpar[name]) ** 2)
        return self.lambda_ewc * loss

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super().__init__(
            ASPP(in_channels, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.aspp = ASPP(in_channels, 256, [6, 12, 18])

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 304 = 256 + 48
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self._init_weight()

    def forward(self, feature, low_level_feature):
        low_level_feature = self.project(low_level_feature)
        
        output_feature = self.aspp(feature)
        output_feature = F.interpolate(
            output_feature, size=low_level_feature.shape[-2:], 
            mode='bilinear', align_corners=False)
        
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabV3PlusSegmentation(nn.Module):
    def __init__(self, backbone_type='resnet50', num_classes=3, output_stride=16, in_channels=4):
        super().__init__()
        
        # 根据output_stride设置dilation rates
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]  # 层2,3,4的dilation设置
            aspp_dilations = [12, 24, 36]  # 更大的dilation rates
        elif output_stride == 16:
            replace_stride_with_dilation = [False, False, True]  # 只在最后一层使用dilation
            aspp_dilations = [6, 12, 18]   # 标准dilation rates
        else:
            raise ValueError('Output stride must be 8 or 16')

        # 初始化backbone
        if backbone_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            # 使用已导入的ResNet模型
            if backbone_type == 'resnet18':
                self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, replace_stride_with_dilation=replace_stride_with_dilation)
            elif backbone_type == 'resnet34':
                self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, replace_stride_with_dilation=replace_stride_with_dilation)
            elif backbone_type == 'resnet50':
                self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, replace_stride_with_dilation=replace_stride_with_dilation)
            elif backbone_type == 'resnet101':
                self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=replace_stride_with_dilation)
            elif backbone_type == 'resnet152':
                self.backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1, replace_stride_with_dilation=replace_stride_with_dilation)
            
            # 修改第一层以支持动态通道数输入
            original_conv = self.backbone.conv1
            original_weights = original_conv.weight.data
            
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            
            # 初始化权重
            with torch.no_grad():
                if in_channels == 3:
                    # 3通道输入，直接使用原始权重
                    self.backbone.conv1.weight.copy_(original_weights)
                elif in_channels == 4:
                    # 4通道输入，复制RGB权重，第4通道使用平均值
                    self.backbone.conv1.weight[:, :3].copy_(original_weights)
                    self.backbone.conv1.weight[:, 3:].copy_(original_weights.mean(dim=1, keepdim=True))
                elif in_channels == 5:
                    # 5通道输入，复制RGB权重，第4、5通道使用平均值
                    self.backbone.conv1.weight[:, :3].copy_(original_weights)
                    mean_weights = original_weights.mean(dim=1, keepdim=True)
                    self.backbone.conv1.weight[:, 3:4].copy_(mean_weights)
                    self.backbone.conv1.weight[:, 4:5].copy_(mean_weights)
                elif in_channels == 1:
                    # 1通道输入，使用RGB权重的平均值
                    self.backbone.conv1.weight.copy_(original_weights.mean(dim=1, keepdim=True))
                else:
                    # 其他通道数，使用插值或重复
                    if in_channels > 3:
                        # 通道数大于3，复制RGB权重并重复
                        self.backbone.conv1.weight[:, :3].copy_(original_weights)
                        for i in range(3, in_channels):
                            self.backbone.conv1.weight[:, i:i+1].copy_(original_weights.mean(dim=1, keepdim=True))
                    else:
                        # 通道数小于3，使用平均值
                        self.backbone.conv1.weight.copy_(original_weights.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: resnet18, resnet34, resnet50, resnet101, resnet152")

        # 设置返回中间层的backbone
        self.backbone = IntermediateLayerGetter(
            self.backbone,
            return_layers={
                'layer1': 'low_level',
                'layer4': 'out'
            }
        )

        # 设置decoder
        self.decoder = DeepLabHeadV3PlusEnhanced(
            in_channels=2048,  # ResNet的输出通道
            low_level_channels=256,  # Layer1的输出通道
            num_classes=num_classes,
            aspp_dilations=aspp_dilations
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # 获取多尺度特征
        features = self.backbone(x)
        
        # decoder处理
        x = self.decoder(features['out'], features['low_level'])
        
        # 上采样到原始大小
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x

class DeepLabHeadV3PlusEnhanced(nn.Module):
    """增强版DeepLabV3+头部"""
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilations):
        super().__init__()
        
        # Low-level特征处理
        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # ASPP模块
        self.aspp = ASPP(in_channels, 256, aspp_dilations)

        # 多级decoder
        self.decoder_stages = nn.ModuleList([
            # 第一级decoder
            nn.Sequential(
                nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 304 = 256 + 48
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            # 第二级decoder（可选）
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        ])

        # 最终分类层
        self.classifier = nn.Conv2d(128, num_classes, 1)
        
        self._init_weight()

    def forward(self, feature, low_level_feature):
        # 处理low-level特征
        low_level_feature = self.low_level_project(low_level_feature)
        
        # ASPP处理
        output_feature = self.aspp(feature)
        
        # 上采样ASPP输出以匹配low-level特征大小
        output_feature = F.interpolate(
            output_feature, 
            size=low_level_feature.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 特征融合
        x = torch.cat([low_level_feature, output_feature], dim=1)
        
        # 多级decoder处理
        for decoder_stage in self.decoder_stages:
            x = decoder_stage(x)
        
        # 最终分类
        return self.classifier(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# --------------------------------------------------
# get_model() Wrapper
# --------------------------------------------------

def count_parameters(model):
    """
    计算模型的参数量
    Args:
        model: PyTorch模型
    Returns:
        total_params: 总参数量（百万）
        trainable_params: 可训练参数量（百万）
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params/1e6, trainable_params/1e6

def get_model(parameters):
    """
    获取DeepLabV3+模型，并根据需要加载预训练权重
    Args:
        parameters: 参数字典
    Returns:
        model: DeepLabV3+模型
    """
    # 获取参数
    backbone_type = parameters.get('Network_backbone_type', 'resnet50')
    output_stride = int(parameters.get('Network_output_stride', 16))
    num_classes = len(parameters.get('Network_classNames', ['Background', 'DeadCell', 'LiveCell']))
    
    # 获取输入通道数
    input_size = parameters.get('Network_InputSize', [224, 224, 4])
    in_channels = int(input_size[2]) if len(input_size) > 2 else 3
    
    print(f"Creating DeepLabV3+ model with:")
    print(f"  - Backbone: {backbone_type}")
    print(f"  - Output stride: {output_stride}")
    print(f"  - Input channels: {in_channels}")
    print(f"  - Number of classes: {num_classes}")
    
    model = DeepLabV3PlusSegmentation(
        backbone_type=backbone_type,
        num_classes=num_classes,
        output_stride=output_stride,
        in_channels=in_channels
    )
    
    # 如果需要加载预训练权重
    if parameters.get('load_PretrainedWeight', '0') == '1':
        try:
            # 使用任务名称作为权重目录
            task_name = parameters.get('Task', 'MicrowellCell4ch_L255D128_k562')
            weight_dir = os.path.join(parameters['project_dir'], 'Networks', task_name)
            best_model_path = get_best_model_path(weight_dir)
            print(f"Loading weights from: {best_model_path}")
            
            # 正确加载权重文件
            checkpoint = torch.load(best_model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 如果是完整的检查点（包含model_state_dict）
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果只保存了模型权重
                model.load_state_dict(checkpoint)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {str(e)}")
            print("Continuing with randomly initialized weights")
    
    return model
