# Standard-library imports for files, JSON, system access, randomness, and memory cleanup.
import os
import json
import sys
import random
import gc
import math
import csv
import re
from contextlib import nullcontext
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
import torch.utils.data as data 
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from full_train import CLIP, ModelConfig as FullTrainModelConfig





ModelConfig = FullTrainModelConfig


@dataclass
class TrainConfig:
    num_epochs: int = 100
    batch_size: int = field(
        default_factory=lambda: int(os.environ.get("MINI_CLIP_BATCH_SIZE", "8"))
    )
    gradient_accumulation_steps: int = field(
        default_factory=lambda: int(
            os.environ.get("MINI_CLIP_GRAD_ACCUMULATION", "8")
        )
    )
    debug: bool = False
    lr_vision: float = 4e-5
    lr_text: float = 1e-5
    lr_proj: float = 1e-3
    betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    weight_decay: float = 0.01
    data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "MINI_CLIP_DATA_DIR",
            os.path.join(os.getcwd(), "flickr8k"),
        )
    )
    ann_file: Optional[str] = None
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    mixed_precision: bool = False

    def __post_init__(self):
        if self.ann_file is None:
            self.ann_file = os.path.join(self.data_dir, "captions.txt")


def clean_caption(caption: str) -> str:
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_ctx(mixed_precision: bool, device: torch.device):
    if mixed_precision and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def tensor_to_device(*tensors, device: torch.device):
    moved = tuple(tensor.to(device, non_blocking=True) for tensor in tensors)
    return moved[0] if len(moved) == 1 else moved


class FLICKR8KDataset(Dataset):
    def __init__(self, root, ann_file, tokenizer, transform=None):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Image directory not found: {root}")
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(f"Captions file not found: {ann_file}")
        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        self.transform = transform
        self.tokenizer = tokenizer
        self.eos = " <|endoftext|>"
        self.ann_file = ann_file

        caps = defaultdict(list)
        with open(ann_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["image"].strip()
                cap = clean_caption(row["caption"].strip())
                if cap and os.path.isfile(os.path.join(root, img)):
                    caps[img].append(cap)

        self.images: List[str] = list(caps.keys())
        self.captions = caps
        if not self.images:
            raise ValueError(
                f"No usable image/caption pairs found in {ann_file} and {root}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.root, img_file)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = random.choice(self.captions[img_file])
        enc = self.tokenizer(
            caption + self.eos,
            add_special_tokens=False,  # add EOS/BOS if defined
            truncation=True,  # cut off if too long
            padding="max_length",  # pad to fixed length
            max_length=76,  # choose your sequence length
            return_tensors="pt",  # return PyTorch tensors
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        return {
            "pixel_values": image,
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "caption": caption,
            "image_id": img_file,
        }
        
def get_flickr8k_dataloader(train_config, model_config, tokenizer=None, **kwargs):
    transform = transforms.Compose(
        [
            transforms.Resize((model_config.vit_img_size, model_config.vit_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # ImageNet mean
                std=[0.5, 0.5, 0.5],  # ImageNet std
            ),
        ]
    )

    dataset = FLICKR8KDataset(
        root=os.path.join(train_config.data_dir, "Images"),
        ann_file=train_config.ann_file,
        tokenizer=tokenizer,
        transform=transform,
    )

    if getattr(train_config, "debug", False):
        dataset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))

    # Defaults similar to standard training loaders
    defaults = dict(
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    defaults.update(kwargs)

    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        **defaults,
    )
    return loader



class CLIPLoss(nn.Module):
    def __init__(self, init_temp=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temp))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        image_embeds: (B, D)
        text_embeds:  (B, D)
        """
        # 1) L2-normalize to get cosine similarities
        img = F.normalize(image_embeds, dim=-1)
        txt = F.normalize(text_embeds, dim=-1)

        # 2) scale = exp(logit_scale); clamp for stability
        scale = self.logit_scale.exp().clamp(1 / 100, 100)

        # 3) logits (B,B); keep in float32 for CE stability
        logits_per_image = (scale * (img @ txt.t())).float()
        logits_per_text = logits_per_image.t()

        # 4) in-batch labels
        target = torch.arange(img.size(0), device=img.device)

        loss_i2t = F.cross_entropy(logits_per_image, target)
        loss_t2i = F.cross_entropy(logits_per_text, target)
        return 0.5 * (loss_i2t + loss_t2i)


def train():
    train_config = TrainConfig()
    model_config = ModelConfig()
    tokenizer = load_tokenizer(model_config.lm_hf_model_name)
    dataloader = get_flickr8k_dataloader(
        train_config,
        model_config,
        tokenizer=tokenizer,
        num_workers=0,
        pin_memory=train_config.device.type == "cuda",
        shuffle=True,
    )

    model = CLIP(model_config).to(train_config.device)
    optimizer = optim.Adam(
        [
            {"params": model.vision_model.parameters(), "lr": train_config.lr_vision},
            {"params": model.text_encoder.parameters(), "lr": train_config.lr_text},
            {
                "params": chain(model.vision_proj.parameters(), model.text_proj.parameters()),
                "lr": train_config.lr_proj,
            },
        ],
        weight_decay=train_config.weight_decay,
    )
    criterion = CLIPLoss(init_temp=1)
    ctx = get_ctx(train_config.mixed_precision, train_config.device)
    mixed_precision = (
        train_config.mixed_precision
        and train_config.device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    if train_config.batch_size < 1:
        raise ValueError("MINI_CLIP_BATCH_SIZE must be at least 1")
    if train_config.gradient_accumulation_steps < 1:
        raise ValueError("MINI_CLIP_GRAD_ACCUMULATION must be at least 1")

    train_loss_per_step = []
    for epoch in range(1, train_config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{train_config.num_epochs}")
        model.train()
        batch_loss = 0.0

        pbar = tqdm(dataloader, desc="Train", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(pbar, start=1):
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pixel_values, input_ids, attention_mask = tensor_to_device(
                pixel_values,
                input_ids,
                attention_mask,
                device=train_config.device,
            )
            with ctx:
                img_embeds, text_embeds = model(
                    pixel_values, input_ids, attention_mask
                )
                loss = criterion(img_embeds, text_embeds)

            scaled_loss = loss / train_config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            running_loss = loss.item()
            train_loss_per_step.append(running_loss)
            batch_loss += running_loss
            pbar.set_postfix(loss=f"{running_loss:.4f}")

            should_step = (
                batch_index % train_config.gradient_accumulation_steps == 0
                or batch_index == len(dataloader)
            )
            if should_step:
                if mixed_precision:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            del img_embeds, text_embeds, loss, scaled_loss
            del pixel_values, input_ids, attention_mask

        print(
            f"Train[{epoch}/{train_config.num_epochs}]: "
            f"loss: {batch_loss / len(dataloader):.4f}"
        )

    sns.set_theme()
    sns.set_context("talk")
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_per_step, label="Step Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Per Step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    train()
