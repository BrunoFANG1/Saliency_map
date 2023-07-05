import os
import re
import ijson
import torch
import CLIP.clip as clip
from torch.utils.data import Dataset
from Util import try_patch_image
from tqdm import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, img_dir):
        self.json_file = json_file
        self.img_dir = img_dir

        with open(self.json_file, 'r') as f:
            self.data = list(ijson.items(f, 'item'))  # load json data into memory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        img_name = obj['image']
        img_caption = re.sub(r'[^\w\s]', '', obj['caption'])
        img_path = os.path.join(self.img_dir, img_name)
        img_name = img_name.split("/")[-1].split(".")[0]
        image = preprocess(Image.open(img_path))
        return image, img_path, img_caption, img_name  # Not really useful, but need to return something


# Usage:
dataset = ImageCaptionDataset("/home/yli556/william/project/dataSet/cc3m/cc3m.json",
                              "/home/yli556/william/project/dataSet/cc3m")

# Use the DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=6, shuffle=False)

# Process the images
for batch in tqdm(dataloader):
    images, img_paths, img_captions, img_names = batch
    try_patch_image(
              model=model,
              device=device,
              imgs=images,
              dirs_name=img_names,
              texts=img_captions,
              save_path="/home/yli556/xfang/Saliency_map/saliency_map"
    )
