import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis

def calculate_dice(preds, targets, num_classes=21):
    dice_scores = []
    for cls in range(num_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection) / union)
    return torch.tensor(dice_scores).mean().item()


def calculate_flops(model):
    model.eval()
    input_tensor = torch.randn(1, 3, 512, 512)
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total()

def save_model(model, directory="model_checkpoints", filename="model.pth"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")    


def plot_metrics(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 1].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[1, 0].plot(history['train_dice'], label='Train Dice')
    axes[1, 0].set_title('Training Dice (Macro-Avg)')
    axes[1, 0].legend()
    axes[1, 1].plot(history['val_dice'], label='Val Dice', color='orange')
    axes[1, 1].set_title('Validation Dice (Macro-Avg)')
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig("Training Report")

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap