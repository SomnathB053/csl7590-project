import os
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import SegmentationDataset
from src.model import SegmentationModel
from src.train import train_model
from src.eval import run_inference
from src.utils import plot_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = os.path.join(os.getcwd(), 'data', 'pascal-voc-2012')
BATCH_SIZE = 16
EPOCHS = 30
full_dataset = SegmentationDataset(root, mode='trainval')
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size], 
                                        generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(
    train_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0, 
    pin_memory=torch.cuda.is_available())
val_loader = DataLoader(
    val_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=torch.cuda.is_available())

model = SegmentationModel()
training_hist = train_model(model= model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            epochs=EPOCHS,
                            lr = 0.001,
                            patience= 5)

plot_metrics(training_hist)

test_dataset = SegmentationDataset(root, mode='test')
test_loader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=torch.cuda.is_available())

run_inference(model= model,
              test_loader=test_loader,
              folder_name= "17_output" )
