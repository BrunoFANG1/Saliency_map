import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import ijson
import re

class CustomImageDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        with open(json_file, 'r') as f:
            objects = ijson.items(f, 'item')
            for obj in objects:
                img_name = obj['image']
                img_caption = re.sub(r'[^\w\s]', '', obj['caption'])
                self.data.append((img_name, img_caption))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data[idx][0])
        image = Image.open(img_name).convert('RGB')
        caption = self.data[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, caption