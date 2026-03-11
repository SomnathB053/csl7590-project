import torch
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import SegmentationDataset
from src.model import SegmentationModel
from src.utils import color_map


def run_inference(model, test_loader, folder_name, model_path='model_checkpoints/model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    output_dir = folder_name
    os.makedirs(output_dir, exist_ok=True)
    palette = color_map(21)
    with torch.no_grad():
        for img_name, images, orig_size in tqdm(test_loader, desc="Inference"):
            name = img_name[0]
            w, h = int(orig_size[0]), int(orig_size[1]) 
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            mask_pil = Image.fromarray(preds).resize((w, h), Image.NEAREST)
            mask_pil.putpalette(palette)
            mask_pil.save(os.path.join(output_dir, f"{name}_mask.png"))
    print(f"Done! Masks saved in {output_dir}")




if __name__=="__main__":
    root = os.path.join(os.getcwd(), 'data', 'pascal-voc-2012')
    test_dataset = SegmentationDataset(root, mode='test')
    test_loader = DataLoader(test_dataset,     batch_size=1,    shuffle=False,    num_workers=0,    pin_memory=torch.cuda.is_available())
    output_folder = input("Enter output folder name: ")
    model = SegmentationModel() # custom segmentation model
    run_inference(model, test_loader, output_folder)

