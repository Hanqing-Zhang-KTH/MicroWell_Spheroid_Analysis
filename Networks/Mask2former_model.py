import os
import re
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
import warnings
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore", message=".*transformers.*")

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
# Mask2Former Model Configuration
# --------------------------------------------------

# Dictionary of available backbone sizes and their corresponding model names
AVAILABLE_BACKBONES = {
    'tiny': {
        'semantic': 'facebook/mask2former-swin-tiny-ade-semantic',
        'instance': 'facebook/mask2former-swin-tiny-coco-instance',
        'panoptic': 'facebook/mask2former-swin-tiny-coco-panoptic'
    },
    'small': {
        'semantic': 'facebook/mask2former-swin-small-ade-semantic',
        'instance': 'facebook/mask2former-swin-small-coco-instance',
        'panoptic': 'facebook/mask2former-swin-small-coco-panoptic'
    },
    'base': {
        'semantic': 'facebook/mask2former-swin-base-ade-semantic',
        'instance': 'facebook/mask2former-swin-base-coco-instance',
        'panoptic': 'facebook/mask2former-swin-base-coco-panoptic'
    },
    'large': {
        'semantic': 'facebook/mask2former-swin-large-ade-semantic',
        'instance': 'facebook/mask2former-swin-large-coco-instance',
        'panoptic': 'facebook/mask2former-swin-large-coco-panoptic'
    }
}

def get_model_name(backbone_size='small', task='semantic'):
    """
    Get the pre-trained model name for the specified backbone size and task.

    Args:
        backbone_size: Size of the backbone ('tiny', 'small', 'base', or 'large')
        task: Task type ('semantic', 'instance', or 'panoptic')

    Returns:
        Model name string for HuggingFace's pretrained models
    """
    if backbone_size not in AVAILABLE_BACKBONES:
        raise ValueError(
            f"Invalid backbone size: {backbone_size}. Available options: {list(AVAILABLE_BACKBONES.keys())}")

    if task not in AVAILABLE_BACKBONES[backbone_size]:
        raise ValueError(f"Invalid task: {task}. Available options: {list(AVAILABLE_BACKBONES[backbone_size].keys())}")

    return AVAILABLE_BACKBONES[backbone_size][task]

def get_modified_mask2former(num_classes=2, pretrained_model_name=None, backbone_size=None, task=None, in_channels=3):
    """
    Create a modified Mask2Former model following the clean reference implementation.

    Args:
        num_classes: Number of segmentation classes (including background, but excluding no-object)
        pretrained_model_name: Name of pretrained model to use as starting point
        backbone_size: Size of the backbone ('tiny', 'small', 'base', or 'large')
        task: Task type ('semantic', 'instance', or 'panoptic')
        in_channels: Number of input channels

    Returns:
        model: Modified Mask2Former model
        image_processor: Mask2Former image processor
    """
    from transformers import Mask2FormerImageProcessor, Mask2FormerConfig
    
    # Get model name
    if pretrained_model_name is None and backbone_size is not None and task is not None:
        pretrained_model_name = get_model_name(backbone_size, task)
    elif pretrained_model_name is None:
        pretrained_model_name = get_model_name('small', 'semantic')

    print(f"Using pretrained model: {pretrained_model_name}")

    # Create image processor
    image_processor = Mask2FormerImageProcessor(
        ignore_index=255, 
        reduce_labels=True
    )

    # First load the config from the pretrained model
    config = Mask2FormerConfig.from_pretrained(pretrained_model_name)

    # Modify config to handle our number of classes
    # Important: Mask2Former adds an extra "no-object" class internally,
    # so config.num_labels should be set to the number of actual classes (including background)
    config.num_labels = num_classes

    # Initialize with modified config (this creates a new model with correct class dimensions)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Handle input channel adaptation if needed
    if in_channels != 3:
        # Get the original first conv layer weights
        original_conv = model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection
        
        print(f"Original conv shape: in={original_conv.in_channels}, out={original_conv.out_channels}")
        
        # Create a new conv layer with correct input channels but same output channels
        new_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize the new layer
        torch.nn.init.xavier_uniform_(new_conv.weight)
        if new_conv.bias is not None:
            torch.nn.init.zeros_(new_conv.bias)
        
        # Copy weights from original layer for the first 3 channels (or min of in_channels and 3)
        with torch.no_grad():
            copy_channels = min(in_channels, 3)
            new_conv.weight[:, :copy_channels, :, :] = original_conv.weight[:, :copy_channels, :, :].clone()
            # Initialize remaining channels with zeros
            if in_channels > 3:
                new_conv.weight[:, 3:, :, :] = 0.0
        
        # Replace the conv layer
        model.model.pixel_level_module.encoder.embeddings.patch_embeddings.projection = new_conv
        
        print(f"Modified input conv layer for {in_channels} channels")

    # Check the class prediction head shape
    original_class_head = model.class_predictor
    in_features = original_class_head.in_features

    # Note: For Mask2Former, class_predictor will have num_classes + 1 outputs
    # The +1 is for the "no-object" class used in the model's internal logic
    real_output_classes = num_classes + 1
    print(f"Creating new class predictor for {num_classes} classes (plus 1 no-object class)")

    # Create a new class prediction head with the correct output dimension
    new_class_head = torch.nn.Linear(in_features, real_output_classes)

    # Initialize the new head
    torch.nn.init.xavier_uniform_(new_class_head.weight)
    torch.nn.init.zeros_(new_class_head.bias)

    # Replace the class prediction head
    model.class_predictor = new_class_head

    print(f"Successfully modified model: {in_channels}-channel input, {num_classes} class output (plus 1 no-object class)")

    return model, image_processor

class Mask2FormerSegmentation(nn.Module):
    """
    Wrapper class for Mask2Former to match DeepLabV3+ interface
    """
    def __init__(self, backbone_type='mask2former', num_classes=2, output_stride=8, in_channels=3, 
                 backbone_size='small', task='semantic', pretrained_model_name=None, **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Create the modified Mask2Former model and image processor
        self.model, self.image_processor = get_modified_mask2former(
            num_classes=num_classes,
            backbone_size=backbone_size,
            task=task,
            in_channels=in_channels,  # Use actual input channels
            pretrained_model_name=pretrained_model_name
        )
        
        # Store original input size for interpolation
        self.original_size = None
        
    def forward(self, x, labels=None):
        # Direct tensor processing to avoid memory-intensive PIL conversion
        # Apply ImageNet normalization directly to tensors
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize images
        x_normalized = (x - mean) / std
        
        if labels is not None:
            # For training: prepare inputs in the format Mask2Former expects
            # Convert labels to the format expected by Mask2Former
            batch_size = x.shape[0]
            
            # Create mask_labels list (one per image in batch)
            mask_labels = []
            for i in range(batch_size):
                # Convert single label tensor to binary masks for each class
                label_tensor = labels[i]  # (H, W)
                num_classes = self.num_classes
                
                # Create binary masks for each class
                binary_masks = []
                for class_id in range(num_classes):
                    binary_mask = (label_tensor == class_id).float()
                    binary_masks.append(binary_mask)
                
                # Stack to create [num_classes, H, W] tensor
                mask_label = torch.stack(binary_masks)
                mask_labels.append(mask_label)
            
            # Create class_labels (list of class IDs for each image)
            class_labels = []
            for i in range(batch_size):
                # All classes are present in each image
                class_ids = torch.arange(num_classes, dtype=torch.long, device=x.device)
                class_labels.append(class_ids)
            
            # Prepare inputs in the format Mask2Former expects
            inputs = {
                "pixel_values": x_normalized,
                "mask_labels": mask_labels,
                "class_labels": class_labels
            }
            
            outputs = self.model(**inputs)
        else:
            # For inference: no labels
            outputs = self.model(pixel_values=x_normalized)
        
        return outputs

# --------------------------------------------------
# get_model() Wrapper
# --------------------------------------------------

def count_parameters(model):
    """
    Calculate the number of parameters in the model
    Args:
        model: PyTorch model
    Returns:
        total_params: Total number of parameters (millions)
        trainable_params: Number of trainable parameters (millions)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params/1e6, trainable_params/1e6

def get_model(parameters):
    """
    Get Mask2Former model and load pretrained weights if needed
    Args:
        parameters: Parameter dictionary
    Returns:
        model: Mask2Former model
    """
    # Get parameters
    backbone_type = parameters.get('Network_backbone_type', 'mask2former')
    output_stride = int(parameters.get('Network_output_stride', 8))
    num_classes = len(parameters.get('Network_classNames', ['Background', 'Fish']))
    
    # Get input channels
    input_size = parameters.get('Network_InputSize', [224, 224, 3])
    # For Mask2Former, always use 3 channels (preprocessing converts input to RGB)
    in_channels = 3
    
    # Get Mask2Former specific parameters
    backbone_size = parameters.get('Mask2Former_backbone_size', 'small')
    task_type = parameters.get('Mask2Former_task_type', 'semantic')
    model_name = parameters.get('Mask2Former_model_name', None)
    
    print(f"Creating Mask2Former model with:")
    print(f"  - Backbone: {backbone_type}")
    print(f"  - Backbone size: {backbone_size}")
    print(f"  - Task type: {task_type}")
    print(f"  - Output stride: {output_stride}")
    print(f"  - Input channels: {in_channels} (preprocessing converts 1ch to 3ch RGB)")
    print(f"  - Number of classes: {num_classes}")
    
    model = Mask2FormerSegmentation(
        backbone_type=backbone_type,
        num_classes=num_classes,
        output_stride=output_stride,
        in_channels=in_channels,
        backbone_size=backbone_size,
        task=task_type,
        pretrained_model_name=model_name
    )
    
    # Load pretrained weights if needed
    if parameters.get('load_PretrainedWeight', '0') == '1':
        try:
            # Use network_dir from parameters (includes Mask2Former suffix if applicable)
            weight_dir = parameters.get('network_dir', os.path.join(parameters['project_dir'], 'Networks', parameters.get('Task', 'VAST_Fish2Classes')))
            best_model_path = get_best_model_path(weight_dir)
            print(f"Loading weights from: {best_model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(best_model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {str(e)}")
            print("Continuing with randomly initialized weights")
    
    return model
