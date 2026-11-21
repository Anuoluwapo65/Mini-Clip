# Import standard-library modules for paths, file management, and subprocesses.
import os
import shutil
import subprocess

# Import PyTorch and its neural-network building blocks.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import typing and utility helpers used by the wider project.
import gc
from typing import List, Optional
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from PIL import Image
import torch.optim as optim
from transformers import AutoTokenizer
import math
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

##### import the vision transformer and language model from Hugging Face Transformers 
from vision_modeling import Vision_Transformer, load_hf_vit_info_scratch 
from language_modeling import TextEncoder, load_from_hf_into_scratch
from modality_projector import Modality_Projector


def get_seed(seed):
    # Set every random source used by preprocessing and model initialization.
    # Seed NumPy so NumPy-generated values are reproducible.
    np.random.seed(seed)
    # Seed PyTorch operations performed on the CPU.
    torch.manual_seed(seed)
    # Check whether a CUDA-capable GPU is available.
    if torch.cuda.is_available():
        # Seed all visible CUDA devices.
        torch.cuda.manual_seed_all(seed)
    else:
        # Request deterministic cuDNN algorithms when CUDA is unavailable.
        torch.backends.cudnn.deterministic = True
        # Disable benchmarking so algorithm selection remains reproducible.
        torch.backends.cudnn.benchmark = False


# Initialize every supported random-number generator with one fixed value.
get_seed(seed=42)

# Select the GPU when available; otherwise use the CPU.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Display the selected execution device.
print("Using this device:", device)


@dataclass
class ModelConfig:
    # Name of the pretrained vision transformer.
    vit_hf_model_name: str = "google/vit-base-patch16-224"
    # Input image side length in pixels.
    vit_img_size:int = 224
    # Side length of one image patch in pixels.
    vit_patch_size:int = 16
    # Width of each vision-transformer hidden vector.
    vit_hidden_size:int = 768
    # Number of vision-transformer layers.
    vit_num_hidden_layers:int = 12
    # Number of attention heads in each vision layer.
    vit_num_attention_heads:int = 12
    # Width of the vision feed-forward intermediate layer.
    vit_intermediate_size:int = 3072
    # Numerical-stability epsilon for vision LayerNorm.
    vit_layer_norm_eps:float = 1e-12
    # Vision-model dropout probability.
    vit_drop_out:float = 0.1

    # Name of the pretrained language model.
    lm_hf_model_name: str = "openai-community/gpt2"
    # Number of token IDs understood by the language model.
    lm_vocab_size:int = 50257
    # Number of transformer blocks.
    lm_n_layer:int = 12
    # Number of attention heads per block.
    lm_n_head :int = 12
    # Width of each token embedding.
    lm_n_embed:int = 768
    # Maximum sequence length supported by positional embeddings.
    lm_n_positions:int = 1024
    # Dropout probability after residual projections.
    lm_resid_pdrop:float = 0.1
    # Dropout probability for embeddings and MLP outputs.
    lm_embd_pdrop:float = 0.1
    # Dropout probability for attention probabilities.
    lm_attn_pdrop:float = 0.1
    # Numerical-stability epsilon for language-model LayerNorm.
    lm_layer_norm_epsilon:float = 1e-5
    # Whether the output vocabulary matrix shares weights with token embeddings.
    lm_tie_word_embeddings:bool = True
    # Token ID marking the end of a sequence.
    lm_eos_token_id:int = 50256
    # Token ID used for padding; the field name is kept for compatibility.
    lm_pad_tokn_id:int = (
        50256
    )

    # Width of a possible vision-to-language projection.
    project_dim :int = 512



def init_weight(module):
    """
    Initialize the weights of the module.
    """
    # Use Xavier initialization for trainable linear projections.
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Use the standard small normal distribution for token embeddings.
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean = 0, std = 0.02)
    # Start normalization layers with an identity transformation.
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
        
        
class CLIP(nn.Module):
    """ 
    CLIP model that combines a vision transformer and a language model.
    """
    
    def __init__(self, config: ModelConfig):
        # Initialize the PyTorch module before registering child modules.
        super().__init__()
        # Keep the architecture and pretrained model names together.
        self.config = config
        # Build the image and text encoders from the shared configuration.
        self.vision_model = Vision_Transformer(config)
        self.text_encoder = TextEncoder(config)
        
        # Map both encoder outputs into the same embedding space.
        self.vision_proj = Modality_Projector(config.vit_hidden_size, config.project_dim)
        self.text_proj = Modality_Projector(config.lm_n_embed, config.project_dim)
        
        # Load pretrained backbone weights before freezing selected layers.
        self._load_pretrained_weights()
        
    def forward(self, image, input_ids, attention_mask:Optional[torch.Tensor] = None):
        # Encode both modalities and return comparable projected embeddings.
        image_embeds = self.encode_image(image)
        text_embdes = self.encode_text(input_ids, attention_mask)
        
        
        return image_embeds, text_embdes
    
    
    def encode_image(self, image:torch.Tensor):
        # Add a batch dimension when the caller provides one image.
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # Extract the vision encoder's image-level representation.
        image_features = self.vision_model(image)
        # Project the representation into the shared CLIP embedding space.
        image_embeds = self.vision_proj(image_features)
        
        return image_embeds
    
    
    
    def encode_text(self, input_ids, attention_mask:Optional[torch.Tensor] = None):
        
        # Add a batch dimension when the caller provides one token sequence.
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        # Extract the end-of-text representation from the text encoder.
        text_features =self.text_encoder(input_ids, attention_mask)
        # Project the representation into the shared CLIP embedding space.
        text_embeds = self.text_proj(text_features)
        
        return text_embeds
    
    
    
    
    def _load_pretrained_weights(self):
        
        # Copy pretrained GPT-2 parameters into the scratch text encoder.
        load_from_hf_into_scratch(self.text_encoder, hf_name = self.config.lm_hf_model_name)
        # Copy pretrained ViT parameters into the scratch vision encoder.
        load_hf_vit_info_scratch(self.vision_model, hf_name = self.config.vit_hf_model_name)
        print(f"Loaded-Vision-Encoder from {self.config.vit_hf_model_name}")
        print(f"Loaded Text Encoder from {self.config.lm_hf_model_name}")
        
        
        # Freeze the configured fraction of each backbone after loading weights.
        self.freeze_partial_backbone()
        
        # Report how much of the complete model can still be optimized.
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}/ {total_params}")
        
        
        
    def freeze_partial_backbone(self, freeze_ratio:float = 0.4):
        # Reject invalid ratios instead of silently freezing the wrong layers.
        if not 0.0 <= freeze_ratio <= 1.0:
            raise ValueError("freeze_ratio must be between 0.0 and 1.0")

        # Freeze the earliest vision-transformer blocks.
        num_layers = len(self.vision_model.blocks)
        num_freeze = int(num_layers * freeze_ratio)
        for layer in self.vision_model.blocks[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
        # Keep the vision positional parameters fixed with the frozen backbone.
        self.vision_model.cls_token.requires_grad = False
        self.vision_model.pos_embed.requires_grad = False
        
        
        # Freeze the earliest language-transformer blocks.
        num_layers = len(self.text_encoder.blocks)
        num_freeze = int(num_layers * freeze_ratio)
        
        for layer in self.text_encoder.blocks[:num_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

        # Freeze token embeddings because they are part of the text backbone.
        for param in self.text_encoder.token_emb.parameters():
            param.requires_grad = False
        
                              
 
 

   