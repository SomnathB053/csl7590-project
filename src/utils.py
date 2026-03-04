import matplotlib.pyplot as plt
import torch

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