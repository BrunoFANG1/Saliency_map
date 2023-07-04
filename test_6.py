import os
import re
import ijson
import torch
from torch.utils.data import Dataset
from Util import try_one_image
from tqdm import tqdm


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
        return img_path, img_caption  # Not really useful, but need to return something


# Usage:
dataset = ImageCaptionDataset("/home/yli556/william/project/dataSet/cc3m/cc3m.json",
                              "/home/yli556/william/project/dataSet/cc3m")

# Use the DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=32, shuffle=False)

# Process the images
for data_batch in tqdm(dataloader):
    img_paths, img_captions = data_batch
    for single_img_path, single_img_caption in zip(img_paths, img_captions):
        print(f"img_path: {single_img_path}")
        try_one_image(img_path=single_img_path, texts=[single_img_caption], save_path="./saliency_map/")

