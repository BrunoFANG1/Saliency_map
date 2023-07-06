import os
import re
import ijson
import json
import torch
import CLIP.clip as clip
from torch.utils.data import Dataset
from Util import try_patch_image
from tqdm import tqdm
from PIL import Image
from PIL import UnidentifiedImageError

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
        try:
            image = preprocess(Image.open(img_path))
        except UnidentifiedImageError:
            print(f"cannot find this image {img_path}, skipping")
            return None
        return image, img_path, img_caption, img_name  # Not really useful, but need to return something

    def generate_json(self, batch_size=64):
        data_dict = {}
    
        # Use DataLoader to handle batching
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=12, shuffle=False)
    
        for batch in dataloader:
            images, img_paths, img_captions, img_names = batch

        # Apply transformations to the images and get saliency words from captions
            salient_word_indices_lists = get_saliency_word(img_captions)
            saliency_maps = get_saliency_maps()

        # Update data_dict
            for img_name, saliency_map_list, salient_word_indices_list in zip(img_names, saliency_maps, salient_word_indices_lists):
                inner_dict = {}
                for salient_word_index, saliency_map in zip(salient_word_indices_list, saliency_map_list):
                    inner_dict[salient_word_index] = saliency_map.tolist()  # Convert tensor to list for JSON serialization

                data_dict[img_name] = inner_dict

    # Write to JSON file
    with open('data.json', 'w') as f:
        json.dump(data_dict, f)


# Usage:
dataset = ImageCaptionDataset("/home/yli556/william/project/dataSet/cc3m/cc3m.json",
                              "/home/yli556/william/project/dataSet/cc3m")

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