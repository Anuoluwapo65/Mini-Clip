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
# Set KAGGLE_KEY and KAGGLE_USERNAME in the environment before running this file.
# PowerShell example: $env:KAGGLE_KEY = "Kaggle_key"
# PowerShell example: $env:KAGGLE_USERNAME = "kaggle_username"
# Allow callers to override the default dataset directory.
DATA_DIR = os.environ.get("MINI_CLIP_DATA_DIR", os.path.join(os.getcwd(), "flickr8k"))
# Store the expected path of the downloaded archive.
ZIP_FILE = os.path.join(DATA_DIR, "flickr8k.zip")


def download_dataset() -> None:
    """Download and extract Flickr8k using the Kaggle CLI."""
    # Refuse to continue when Kaggle authentication variables are absent.
    if not os.environ.get("KAGGLE_KEY") or not os.environ.get("KAGGLE_USERNAME"):
        raise RuntimeError(
            "Set KAGGLE_KEY and KAGGLE_USERNAME before running this script."
        )
    # Refuse to continue when the Kaggle command-line program is unavailable.
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The Kaggle CLI is not installed or is not on PATH. "
            "Install it with: python -m pip install kaggle"
        )

    # Create the destination directory if it does not already exist.
    os.makedirs(DATA_DIR, exist_ok=True)
    # Ask Kaggle to download the Flickr8k dataset into the destination directory.
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "adityajn105/flickr8k",
            "--path",
            DATA_DIR,
        ],
        check=True,
    )
    # Confirm that the download created the archive expected by this script.
    if not os.path.isfile(ZIP_FILE):
        raise FileNotFoundError(f"Kaggle download did not create {ZIP_FILE}")

    # Import zipfile only when extraction is actually needed.
    import zipfile

    # Open the downloaded archive and extract all of its contents.
    with zipfile.ZipFile(ZIP_FILE) as archive:
        archive.extractall(DATA_DIR)


# Download data only when this module is executed directly.
if __name__ == "__main__":
    download_dataset()
# Store the image directory used by the vision dataset.
ROOT = os.path.join(DATA_DIR, "Images")
# Store the captions file used by the data pipeline.
ANN_FILE = os.path.join(DATA_DIR, "captions.txt")






def get_seed(seed):
    # Seed NumPy operations for reproducible preprocessing.
    np.random.seed(seed)
    # Seed CPU-side PyTorch operations.
    torch.manual_seed(seed)
    # Check whether CUDA is available before using CUDA-specific functions.
    if torch.cuda.is_available():
        # Seed every visible CUDA device.
        torch.cuda.manual_seed_all(seed)
    else:
        # Request deterministic cuDNN behavior on CPU-only systems.
        torch.backends.cudnn.deterministic = True
        # Allow cuDNN to benchmark available algorithms.
        torch.backends.cudnn.benchmark = True


# Use one fixed seed for repeatable experiments.
get_seed(seed=42)

# Select a GPU when available and otherwise use the CPU.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Display the selected execution device.
print("Using this device:", device)





@dataclass
class ModelConfig:
    # Name of the pretrained vision transformer.
    vit_hf_model_name: str = "google/vit-base-patch16-224"
    # Input image side length in pixels.
    vit_img_size:int = 224
    # Side length of each square image patch.
    vit_patch_size:int = 16
    # Width of each vision-transformer hidden vector.
    vit_hidden_size:int = 768
    # Number of transformer layers in the vision encoder.
    vit_num_hidden_layers:int = 12
    # Number of attention heads per transformer layer.
    vit_num_attention_heads:int = 12
    # Width of the feed-forward intermediate layer.
    vit_intermediate_size:int = 3072
    # LayerNorm numerical-stability value.
    vit_layer_norm_eps:float = 1e-12
    # Dropout probability used throughout the vision encoder.
    vit_drop_out:float = 0.1

    # Name of the language model used by the companion text encoder.
    lm_hf_model_name: str = "openai-community/gpt2-medium"
    # Vocabulary size used by the companion language model.
    lm_vocab_size:int = 5027
    # Number of language-model transformer blocks.
    lm_n_layer:int = 24
    # Number of language-model attention heads.
    lm_n_head :int = 16
    # Language-model embedding width.
    lm_n_embed:int = 1024
    # Maximum language-model sequence length.
    lm_n_positions:int = 76
    # Language-model residual dropout probability.
    lm_resid_pdrop:float = 0.1
    # Language-model embedding dropout probability.
    lm_embd_pdrop:float = 0.1
    # Language-model attention dropout probability.
    lm_attn_pdrop:float = 0.1
    # Language-model LayerNorm epsilon.
    lm_layer_norm_epsilon:float = 1e-5
    # Whether language input and output embeddings share weights.
    lm_tie_word_embeddings:bool = True
    # End-of-sequence token identifier.
    lm_eos_token_id:int = 50256
    # Padding token identifier.
    lm_pad_tokn_id: int = (
        50256
    )

    # Optional projection width shared by image and text representations.
    project_dim :int = 512


class MLP(nn.Module):
    # Build the feed-forward sublayer used inside a vision transformer block.
    def __init__(self, config:ModelConfig):
        # Initialize the parent PyTorch module.
        super(MLP, self).__init__()
        # Expand the hidden representation to the intermediate width.
        self.fc1 = nn.Linear(config.vit_hidden_size, config.vit_intermediate_size)
        # Use the GELU activation used by ViT.
        self.act = nn.GELU()
        # Project the intermediate representation back to hidden width.
        self.fc2 = nn.Linear(config.vit_intermediate_size, config.vit_hidden_size)
        # Regularize the MLP output.
        self.drop = nn.Dropout(config.vit_drop_out)

    # Apply the two linear layers and activation to a token sequence.
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        # Expand, activate, contract, and then drop out features.
        return(self.drop(self.fc2(self.act(self.fc1(x)))))


class Attention(nn.Module):
    # Build multi-head self-attention for image patch tokens.
    def __init__(self, config:ModelConfig):
        # Initialize the parent PyTorch module.
        super(Attention, self).__init__()
        # Store the number of attention heads.
        self.num_heads = config.vit_num_attention_heads
        # Compute the width assigned to each attention head.
        self.head_dim = config.vit_hidden_size // config.vit_num_attention_heads
        # Scale attention logits by the inverse square root of head width.
        self.scale = self.head_dim ** -0.5

        # Project hidden states into queries.
        self.q_proj = nn.Linear(config.vit_hidden_size, config.vit_hidden_size)
        # Project hidden states into keys.
        self.k_proj = nn.Linear(config.vit_hidden_size, config.vit_hidden_size)
        # Project hidden states into values.
        self.v_proj = nn.Linear(config.vit_hidden_size, config.vit_hidden_size)

        # Project combined head outputs back to hidden width.
        self.out_proj = nn.Linear(config.vit_hidden_size, config.vit_hidden_size)
        # Drop attention probabilities during training.
        self.attn_dropout = nn.Dropout(config.vit_drop_out)
        # Drop projected output features during training.
        self.proj_drop_out = nn.Dropout(config.vit_drop_out)

    # Compute multi-head self-attention for all image tokens.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read batch size, token count, and hidden width.
        B, N, C = x.shape
        # Project and reshape queries into separate heads.
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Project and reshape keys into separate heads.
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Project and reshape values into separate heads.
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled query-key similarity scores.
        attn = q @ k.transpose(-2, -1) * self.scale
        # Normalize scores into probabilities across key tokens.
        attn = attn.softmax(dim = -1)
        # Apply attention dropout during training.
        attn = self.attn_dropout(attn)

        # Mix value vectors using the attention probabilities.
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Project the mixed heads back into the model hidden space.
        out = self.out_proj(out)
        # Apply output dropout and return the attention result.
        return self.proj_drop_out(out)
    
    
    
    
class Block(nn.Module):
    # Construct one pre-normalized vision-transformer block.
    def __init__(self, config:ModelConfig):
        # Initialize the parent module.
        super(Block, self).__init__()
        # Normalize tokens before self-attention.
        self.ln1 = nn.LayerNorm(config.vit_hidden_size, eps = config.vit_layer_norm_eps)
        # Create the self-attention sublayer.
        self.attn = Attention(config)
        # Normalize tokens before the feed-forward sublayer.
        self.ln2 = nn.LayerNorm(config.vit_hidden_size, eps = config.vit_layer_norm_eps)
        # Create the feed-forward sublayer.
        self.mlp = MLP(config)

    # Apply attention and MLP transformations with residual connections.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add the attention output to the original token sequence.
        x = x + self.attn(self.ln1(x))
        # Add the MLP output to the updated token sequence.
        x = x + self.mlp(self.ln2(x))
        # Return the transformed token sequence.
        return x
    
    
    
    
    
class Vision_Transformer(nn.Module):
    # Construct a Vision Transformer that returns its class-token embedding.
    def __init__(self, config:ModelConfig):
        # Initialize the parent module.
        super(Vision_Transformer, self).__init__()

        # Keep the configuration available to the weight-loader.
        self.config = config
        # Convert non-overlapping image patches into hidden vectors.
        self.patch_embed = nn.Conv2d(
            3, config.vit_hidden_size, kernel_size = config.vit_patch_size, stride = config.vit_patch_size
        )
        # Calculate how many patches fit into one configured image.
        num_patches = (config.vit_img_size // config.vit_patch_size) ** 2
        # Create the learnable classification token.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vit_hidden_size))
        # Create positional vectors for the class token and every patch.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.vit_hidden_size))
        # Regularize the token-plus-position sequence.
        self.pos_dropout = nn.Dropout(config.vit_drop_out)

        # Stack the configured number of transformer blocks.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.vit_num_hidden_layers)])
        # Normalize the final token sequence.
        self.norm = nn.LayerNorm(config.vit_hidden_size, eps = config.vit_layer_norm_eps)

    # Convert an image batch into one representation per image.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read the batch size.
        B = x.shape[0]
        # Convert the image into a grid of patch embeddings.
        x = self.patch_embed(x)
        # Flatten the patch grid and make patches the sequence dimension.
        x = x.flatten(2).transpose(1, 2)
        # Expand the learned class token once per image.
        cls = self.cls_token.expand(B, -1, -1)
        # Put the class token before the patch-token sequence.
        x = torch.cat([cls, x], dim = 1)
        # Add positional information to each token.
        x = x + self.pos_embed
        # Apply positional dropout.
        x = self.pos_dropout(x)

        # Process the sequence through every transformer block.
        for blk in self.blocks:
            x = blk(x)
        # Normalize the final sequence.
        x = self.norm(x)

        # Select the first token as the image-level representation.
        cls_out = x[:, 0]
        # Return one representation per image.
        return cls_out
    
    
@torch.no_grad()
def load_hf_vit_info_scratch(
    model:Vision_Transformer,
    hf_name:str = "google/vit-base-patch16-224",
    device:Optional[str] = None,
):
    # Import Hugging Face classes only when pretrained loading is requested.
    from transformers import AutoConfig, ViTModel

    # Read the architecture configuration of the requested pretrained model.
    hf_cfg = AutoConfig.from_pretrained(hf_name, add_pooling_layer = False)
    # Confirm that the scratch model has compatible dimensions.
    assert hf_cfg.hidden_size == model.config.vit_hidden_size
    assert hf_cfg.num_hidden_layers == model.config.vit_num_hidden_layers
    assert hf_cfg.num_attention_heads == model.config.vit_num_attention_heads
    assert hf_cfg.intermediate_size == model.config.vit_intermediate_size
    assert hf_cfg.image_size == model.config.vit_img_size
    assert hf_cfg.patch_size == model.config.vit_patch_size

    # Load the pretrained ViT weights.
    hf_model = ViTModel.from_pretrained(hf_name)
    # Extract the pretrained state dictionary.
    hf_sd = hf_model.state_dict()
    # Extract the scratch model state dictionary for in-place replacement.
    sd = model.state_dict()

    # Copy the patch projection weights and bias.
    sd["patch_embed.weight"] = hf_sd["embeddings.patch_embeddings.projection.weight"]
    sd["patch_embed.bias"] = hf_sd["embeddings.patch_embeddings.projection.bias"]

    # Copy the class token and positional embeddings.
    sd["cls_token"] = hf_sd["embeddings.cls_token"]
    sd["pos_embed"] = hf_sd["embeddings.position_embeddings"]

    # Copy the parameters of every transformer block.
    for i in range(model.config.vit_num_hidden_layers):
        # Build the scratch-model and Hugging Face parameter prefixes.
        p = f"blocks.{i}"
        h = f"encoder.layer.{i}"

        # Copy the first LayerNorm and all attention projections.
        
        sd[f"{p}.ln1.weight"] =hf_sd[f"{h}.layernorm_before.weight"]
        sd[f"{p}.ln1.bias"] = hf_sd[f"{h}.layernorm_before.bias"]
        sd[f"{p}.attn.q_proj.weight"] = hf_sd[f"{h}.attention.attention.query.weight"]
        sd[f"{p}.attn.q_proj.bias"] = hf_sd[f"{h}.attention.attention.query.bias"]
        sd[f"{p}.attn.k_proj.weight"] = hf_sd[f"{h}.attention.attention.key.weight"]
        sd[f"{p}.attn.k_proj.bias"] = hf_sd[f"{h}.attention.attention.key.bias"]
        sd[f"{p}.attn.v_proj.weight"] = hf_sd[f"{h}.attention.attention.value.weight"]
        sd[f"{p}.attn.v_proj.bias"] = hf_sd[f"{h}.attention.attention.value.bias"]
        # Copy the second LayerNorm and both MLP projections.
        sd[f"{p}.ln2.weight"] = hf_sd[f"{h}.layernorm_after.weight"]
        sd[f"{p}.ln2.bias"] = hf_sd[f"{h}.layernorm_after.bias"]
        sd[f"{p}.mlp.fc1.weight"] = hf_sd[f"{h}.intermediate.dense.weight"]
        sd[f"{p}.mlp.fc1.bias"] = hf_sd[f"{h}.intermediate.dense.bias"]
        sd[f"{p}.mlp.fc2.weight"] = hf_sd[f"{h}.output.dense.weight"]
        sd[f"{p}.mlp.fc2.bias"] = hf_sd[f"{h}.output.dense.bias"]

    # Copy the final LayerNorm parameters.
    sd["norm.weight"] = hf_sd["layernorm.weight"]
    sd["norm.bias"] = hf_sd["layernorm.bias"]
    # Load the copied values and allow PyTorch to report unmatched names.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Move the model to the requested device when provided.
    if device is not None:
        model.to(device)
    # Confirm that loading completed.
    print("Loaded HF ViT weights.")
    # Report scratch-model parameters that were not filled.
    if missing:
        print("Missing keys:", missing)
    # Report pretrained parameters that were not consumed.
    if unexpected:
        print("Unexpected keys:", unexpected)
    # Return the model containing the imported parameters.
    return model


      




        














        

        
        
        
    
    
    
            
            
            
        
        
        
    


    
        
        
    
    
