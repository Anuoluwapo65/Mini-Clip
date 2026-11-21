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


class Modality_Projector(nn.Module):
    def __init__(self, in_dim, out_dim, drop_out = 0.1):
        super(Modality_Projector, self).__init__()
        
        self.projection = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(drop_out)
        self.layer_norm = nn.LayerNorm(out_dim)
        
        self._init_weights()
        
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        residual = projected
        projected = self.gelu(projected)
        projected = self.fc(projected)
        projected = self.dropout(projected)
        return self.layer_norm(projected + residual)
        