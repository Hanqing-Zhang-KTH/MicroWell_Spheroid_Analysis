import os
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import numpy as np

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
# 3D UNet Model
# --------------------------------------------------

class UNet3DSegmentation(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        
        # Extract parameters
        self.in_channels = int(parameters['Network_InputSize'][3])  # Number of channels
        self.num_classes = len(parameters['Network_classNames'])
        self.output_stride = int(parameters.get('Network_output_stride', 8))
        
        # UNet configuration from parameters
        channels = parameters.get('Network_channels', [32, 64, 128, 256, 512])
        if isinstance(channels, str):
            channels = [int(x) for x in channels.split(',')]
        else:
            channels = [int(x) for x in channels]
        
        strides = parameters.get('Network_strides', [2, 2, 2, 2])
        if isinstance(strides, str):
            strides = [int(x) for x in strides.split(',')]
        else:
            strides = [int(x) for x in strides]
        
        num_res_units = int(parameters.get('Network_num_res_units', 2))
        dropout = float(parameters.get('Network_dropout', 0.1))

        # Create MONAI UNet
        self.unet = UNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weight()
        
    def _init_weight(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            Output tensor of shape (B, num_classes, D, H, W)
        """
        return self.unet(x)

def count_parameters(model):
    """Count the number of trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return total_params

def get_model(parameters):
    """
    Factory function to create 3D UNet model
    Args:
        parameters: Configuration parameters
    Returns:
        model: Configured 3D UNet model
    """
    model = UNet3DSegmentation(parameters)
    count_parameters(model)
    return model

def load_pretrained_weights(model, weight_path, device):
    """
    Load pretrained weights for the model
    Args:
        model: The model to load weights into
        weight_path: Path to the pretrained weights
        device: Device to load the model on
    """
    if not os.path.exists(weight_path):
        print(f"Warning: Pretrained weight file not found: {weight_path}")
        return False
    
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model state dict from {weight_path}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"Loaded state dict from {weight_path}")
            else:
                # Assume the entire checkpoint is the state dict
                model.load_state_dict(checkpoint)
                print(f"Loaded weights from {weight_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded weights from {weight_path}")
        
        return True
        
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        return False 