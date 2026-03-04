import torch
import torch.nn as nn
from tqdm import tqdm
import os

from utils import calculate_dice

def train_model(model, device, train_loader, val_loader, epochs=30, lr=1e-3, patience=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=255) # 255 is the uncertain pixel index
    
    checkpoint_dir = 'model_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    epoch_pbar = tqdm(range(epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        model.train()
        t_loss, t_dice = 0, 0
        train_pbar = tqdm(train_loader, desc="Training", position=1, leave=False)
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            t_dice += calculate_dice(preds, masks)
            train_pbar.set_description(f"Batch Loss: {loss.item():.4f}")
        model.eval()
        v_loss, v_dice = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                v_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                v_dice += calculate_dice(preds, masks)
        
        # average loss this epoch
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        avg_t_dice = t_dice / len(train_loader)
        avg_v_dice = v_dice / len(val_loader)
        
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['train_dice'].append(avg_t_dice)
        history['val_dice'].append(avg_v_dice)
        
        epoch_pbar.set_description(f"Epoch {epoch} | TrainLoss: {avg_t_loss:.3f} | ValLoss: {avg_v_loss:.3f} | ValDice: {avg_v_dice:.3f}")

        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'segmodel.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    print("Training complete!")          
    return history