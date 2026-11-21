# Standard-library imports for files, JSON, system access, randomness, and memory cleanup.
import os
import json
import sys
import random
import gc
import math
from itertools import chain
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

# Third-party imports used by the training and visualization pipeline.
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Dataset, tokenizer, and trainer utilities imported from PyTorch and Transformers.
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer



def get_seed(seed):
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
        # Allow cuDNN to benchmark algorithms for the selected hardware.
        torch.backends.cudnn.benchmark = True


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
    lm_hf_model_name: str = "openai-community/gpt2-medium"
    # Number of token IDs understood by the language model.
    lm_vocab_size:int = 5027
    # Number of transformer blocks.
    lm_n_layer:int = 24
    # Number of attention heads per block.
    lm_n_head :int = 16
    # Width of each token embedding.
    lm_n_embed:int = 1024
    # Maximum sequence length supported by positional embeddings.
    lm_n_positions:int = 76
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





class GELU(nn.Module):
    # Define the activation module used by GPT-style transformer blocks.
    def forward(self, x):
        # Apply GELU, which GPT-2 uses instead of the ReLU activation.
        return F.gelu(x)
    
    
    
class CasualSelfAttention(nn.Module):
    # Construct one causal multi-head self-attention layer.
    def __init__(self, config:ModelConfig):
        # Initialize the parent PyTorch module.
        super(CasualSelfAttention, self).__init__()
        # Require an embedding width that divides evenly across attention heads.
        assert config.lm_n_embed % config.lm_n_head == 0, "Embedding dimension must be divisible by number of heads"
        # Save the number of attention heads.
        self.n_head = config.lm_n_head
        # Compute the width assigned to each attention head.
        self.head_dim = config.lm_n_embed // config.lm_n_head
        # Project each token into concatenated query, key, and value vectors.
        self.c_attn = nn.Linear(config.lm_n_head * self.head_dim, 3 * config.lm_n_head * self.head_dim)
        # Project the combined attention output back to model width.
        self.c_proj = nn.Linear(config.lm_n_embed, config.lm_n_embed, bias = True)
        # Create dropout for attention values.
        self.attn_drop = nn.Dropout(config.lm_attn_pdrop)
        # Create dropout for residual outputs.
        self.resid_drop = nn.Dropout(config.lm_resid_pdrop)

        # Build a lower-triangular mask that blocks future tokens.
        mask = torch.tril(torch.ones(config.lm_n_positions, config.lm_n_positions), 1).bool()
        # Register the mask so it moves with the model without being saved.
        self.register_buffer("casualmask", mask, persistent = False)

    # Compute causal attention for a batch of hidden-state sequences.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read batch size, sequence length, and embedding width.
        B,T, C = x.size()
        # Produce concatenated query, key, and value projections.
        qkv = self.c_attn(x)
        # Split the last dimension into query, key, and value tensors.
        q,k, v = qkv.split(C, dim = -1)
        # Reshape queries into batch, head, time, and head-width dimensions.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape keys into batch, head, time, and head-width dimensions.
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape values into batch, head, time, and head-width dimensions.
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Calculate scaled query-key similarity scores.
        attn = ( q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply the causal mask before converting scores to probabilities.
        attn = attn.masked_fill(self.casualmask[:T, :T] == 0, float("-inf"))
        # Convert masked scores into probabilities across the key positions.
        attn = F.softmax(attn, dim = -1)
        # Randomly drop some attention probabilities during training.
        attn = self.attn_drop(attn)
        # Combine value vectors according to the attention probabilities.
        y = attn @ v
        # Restore batch, sequence, and embedding dimensions.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Project the attention output and apply residual dropout.
        Y = self.resid_drop(self.c_proj(y))
        # Return the transformed hidden states.
        return Y
    
    
class TMLP(nn.Module):
    # Construct the feed-forward sublayer used after attention.
    def __init__(self, config:ModelConfig):
        # Initialize the parent module.
        super(TMLP, self).__init__()
        # Expand each token representation to four times the model width.
        self.fc = nn.Linear(config.lm_n_embed, 4* config.lm_n_embed, bias = True)
        # Use the GPT-style GELU non-linearity.
        self.act = nn.GELU()
        # Project the expanded representation back to model width.
        self.fc2 = nn.Linear( 4 * config.lm_n_embed,config.lm_n_embed, bias = True)
        # Regularize the feed-forward output.
        self.drop = nn.Dropout(config.lm_embd_pdrop)

    # Run one token sequence through the feed-forward sublayer.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the final embedding dimension.
        x = self.fc(x)
        # Apply the activation function.
        x = self.act(x)
        # Return to the original embedding dimension.
        x = self.fc2(x)
        # Apply output dropout.
        x = self.drop(x)

        # Return the transformed sequence.
        return x


class TBLOCK(nn.Module):
    # Construct one pre-normalized transformer block.
    def __init__(self, config:ModelConfig):
        # Initialize the parent module.
        super(TBLOCK, self).__init__()
        # Normalize inputs before self-attention.
        self.ln1 = nn.LayerNorm(config.lm_n_embed,eps = config.lm_layer_norm_epsilon)
        # Create the causal self-attention sublayer.
        self.attn = CasualSelfAttention(config)
        # Normalize the attention result before the MLP.
        self.ln2 = nn.LayerNorm(config.lm_n_embed, eps = config.lm_layer_norm_epsilon)
        # Create the feed-forward sublayer.
        self.mlp = TMLP(config)

    # Apply attention and MLP sublayers with residual connections.
    def forward(self, x: torch.Tensor, attention_mask=None)-> torch.Tensor:
        # Add the attention transformation to the original input.
        x = x + self.attn(self.ln1(x))
        # Add the feed-forward transformation to the updated input.
        x = x + self.mlp(self.ln2(x))

        # Return the block output.
        return x


class TextEncoder(nn.Module):
    # Construct the token-and-position embedding transformer.
    def __init__(self, config:ModelConfig):
        # Initialize the parent module.
        super(TextEncoder, self).__init__()
        # Retain the configuration for later token selection.
        self.config = config
        # Create a lookup table for token embeddings.
        self.token_emb = nn.Embedding(config.lm_vocab_size, config.lm_n_embed)
        # Create a lookup table for position embeddings.
        self.pos_embed = nn.Embedding(config.lm_n_positions,config.lm_n_embed)       
        # Drop out the combined token and position embeddings.
        self.drop_out = nn.Dropout(config.lm_embd_pdrop)
        # Stack the configured number of transformer blocks.
        self.blocks = nn.ModuleList([TBLOCK(config)for _ in range(config.lm_n_layer)])
        # Normalize the final hidden states.
        self.ln_f = nn.LayerNorm(config.lm_n_embed,eps = config.lm_layer_norm_epsilon)
        # Map hidden states to vocabulary logits.
        self.lm_head = nn.Linear(config.lm_n_embed, config.lm_vocab_size, bias = False)
        # Optionally tie output weights to the input embedding weights.
        if config.lm_tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight
        # Initialize every supported layer with the custom initializer below.
        self.apply(self._init_weights)

    # Initialize linear and embedding layer parameters.
    def _init_weights(self, module):
        # Initialize linear weights from a small normal distribution.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            # Initialize a linear bias when that layer has one.
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize embedding weights from the same normal distribution.
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    # Build position indices without tracking gradients.
    @torch.no_grad()
    def pos_indices(self,T, device):
        # Return positions shaped for broadcasting over the batch.
        return torch.arange(T, device = device).unsqueeze(0)

    # Convert token IDs into one representation per input sequence.
    def forward(self, input_ids, attention_mask:Optional[torch.Tensor] = None):
        # Read batch size and sequence length from the token matrix.
        B, T = input_ids.shape
        # Keep new tensors on the same device as the input IDs.
        device = input_ids.device

        # Create position IDs from zero through the final input position.
        pos = self.pos_indices(T, device)
        # Add token and position embeddings element by element.
        x = self.token_emb(input_ids) + self.pos_embed(pos)
        # Apply embedding dropout.
        x = self.drop_out(x)

        # Pass the sequence through each transformer block.
        for blk in self.blocks:
            x = blk(x, attention_mask)
        # Normalize the final sequence representations.
        x = self.ln_f(x)

        # Find the final end-of-sequence token in each example.
        eot_idx = (input_ids == self.config.lm_eos_token_id).int().argmax(dim=1)
        # Gather the hidden state at each example's selected end position.
        eot_rep = x[torch.arange(B, device = device), eot_idx, :]

        # Return one representation per input sequence.
        return eot_rep
    
    
    
def load_from_hf_into_scratch(
    model:TextEncoder,
    hf_name:str = "openai-community/gpt2",
    device:Optional[str] = None,
):
    # Import the configuration loader and GPT-2 implementation locally.
    from transformers import AutoConfig, GPT2LMHeadModel

    # Transpose Hugging Face Conv1D weights only when the target shape requires it.
    def maybe_T(src:torch.Tensor, target_shape:torch.Size):
        # Return the source unchanged when its shape already matches.
        if src.shape == target_shape:
            return src
        # Transpose two-dimensional weights when the dimensions are reversed.
        if src.shape[::-1] == tuple(target_shape):
            return src.t()
        # Preserve unexpected shapes so load_state_dict reports them explicitly.
        return src

    # Read the architecture settings from the requested Hugging Face model.
    hf_cfg = AutoConfig.from_pretrained(hf_name)
    # Verify that the scratch model has compatible transformer dimensions.
    assert hf_cfg.n_layer == model.config.lm_n_layer
    assert hf_cfg.n_head == model.config.lm_n_head
    assert hf_cfg.n_embd == model.config.lm_n_embed
    assert hf_cfg.vocab_size == model.config.lm_vocab_size
    assert hf_cfg.n_positions == model.config.lm_n_positions
    assert hf_cfg.resid_pdrop == model.config.lm_resid_pdrop
    assert hf_cfg.embd_pdrop == model.config.lm_embd_pdrop
    assert hf_cfg.attn_pdrop == model.config.lm_attn_pdrop
    assert hf_cfg.layer_norm_epsilon == model.config.lm_layer_norm_epsilon

    # Download or load the pretrained GPT-2 weights.
    hf_model = GPT2LMHeadModel.from_pretrained(hf_name)
    # Extract the pretrained parameter dictionary.
    hf_sd = hf_model.state_dict()
    # Extract the scratch model parameter dictionary.
    sd = model.state_dict()

    # Copy token embeddings from Hugging Face into the scratch model.
    sd["token_emb.weight"] = hf_sd["transformer.wte.weight"]
    # Read the pretrained positional embedding matrix.
    pos_src = hf_sd["transformer.wpe.weight"]
    # Truncate positional embeddings when the scratch model is shorter.
    if pos_src.size(0) >= model.config.lm_n_positions:
        sd["pos_embed.weight"] = pos_src[: model.config.lm_n_positions]
    # Pad positional embeddings when the scratch model is longer.
    else:
        pad = model.config.lm_n_positions - pos_src.size(0)
        sd["pos_embed.weight"] = torch.cat([pos_src, pos_src[-1:].repeat(pad, 1)], dim=0)

    # Use the scratch model's embedding width for shape calculations.
    C = model.config.lm_n_embed
    # Visit every transformer block and copy its corresponding weights.
    for i in range(model.config.lm_n_layer):
        # Build the scratch-model and Hugging Face parameter prefixes.
        p = f"blocks.{i}"
        h = f"transformer.h.{i}"

        # Copy both LayerNorm parameter sets.
        sd[f"{p}.ln1.weight"] = hf_sd[f"{h}.ln_1.weight"]
        sd[f"{p}.ln1.bias"] = hf_sd[f"{h}.ln_1.bias"]
        sd[f"{p}.ln2.weight"] = hf_sd[f"{h}.ln_2.weight"]
        sd[f"{p}.ln2.bias"] = hf_sd[f"{h}.ln_2.bias"]

        # Copy fused query-key-value and attention-output projections.
        sd[f"{p}.attn.c_attn.weight"] = maybe_T(hf_sd[f"{h}.attn.c_attn.weight"], torch.Size([3 * C, C]))
        sd[f"{p}.attn.c_attn.bias"] = hf_sd[f"{h}.attn.c_attn.bias"]
        sd[f"{p}.attn.c_proj.weight"] = maybe_T(hf_sd[f"{h}.attn.c_proj.weight"], torch.Size([C, C]))
        sd[f"{p}.attn.c_proj.bias"] = hf_sd[f"{h}.attn.c_proj.bias"]

        # Copy the first MLP projection into the scratch layer's ``fc``.
        sd[f"{p}.mlp.fc.weight"] = maybe_T(hf_sd[f"{h}.mlp.c_fc.weight"], torch.Size([4 * C, C]))
        sd[f"{p}.mlp.fc.bias"] = hf_sd[f"{h}.mlp.c_fc.bias"]
        # Copy the second MLP projection into the scratch layer's ``fc2``.
        sd[f"{p}.mlp.fc2.weight"] = maybe_T(hf_sd[f"{h}.mlp.c_proj.weight"], torch.Size([C, 4 * C]))
        sd[f"{p}.mlp.fc2.bias"] = hf_sd[f"{h}.mlp.c_proj.bias"]

    # Copy the final transformer LayerNorm.
    sd["ln_f.weight"] = hf_sd["transformer.ln_f.weight"]
    sd["ln_f.bias"] = hf_sd["transformer.ln_f.bias"]

    # Copy the output head only when embeddings are not tied.
    if not model.config.lm_tie_word_embeddings:
        if "lm_head.weight" in hf_sd:
            sd["lm_head.weight"] = hf_sd["lm_head.weight"]
        else:
            sd["lm_head.weight"] = hf_sd["transformer.wte.weight"]

    # Load copied parameters while allowing intentionally omitted tied weights.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Move the completed model when a target device was supplied.
    if device is not None:
        model.to(device)

    # Report the completed conversion.
    print("Loaded HF weights with Conv1D->Linear transposes.")
    # Report parameters that were not found in the scratch model.
    if missing:
        print("Missing keys:", missing)
    # Report parameters from Hugging Face that were not consumed.
    if unexpected:
        print("Unexpected keys:", unexpected)
    # Return the model with the imported parameters.
    return model
    