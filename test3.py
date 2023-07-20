import os
import tsv
import re
import json
from PIL import Image
import torch
import CLIP.clip as clip
from torch.utils.data import Dataset
from Util import get_saliency_word, get_saliency_map, clip_text
from tqdm import tqdm
from PIL import Image
from PIL import UnidentifiedImageError

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        name, _ = os.path.splitext(image_file)
        try:
            img = Image.open(os.path.join(self.data_dir, image_file))
        except (IOError, SyntaxError) as e:
            print(f'Error opening image file {image_file}:{e}')
            return torch.zeros(3, 224, 224), 'This is a junk'
        
        caption_file = os.path.join(self.data_dir, image_file.replace('.png', '.txt'))
        if not os.path.exists(caption):
            print(f'Caption file does not exist: {caption_file}')
            return torch.zeros(3, 224, 224), 'This is a junk'
        try:
            with open(caption_file, 'r') as f:
                caption = f.read().strio()
                caption = re.sub(r'[^\w\s]', '', caption])
        except IOError as e:
            print(f'Error opening caption file {caption_file}:{e}')
            return torch.zeros(3, 224, 224), 'This is a junk'
        return img, caption, name


    def generate_json(self, batch_size=128):
        outer_dict = {}
    
        # Use DataLoader to handle batching
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=4, shuffle=False)
    
        for batch in tqdm(dataloader):
            images, img_captions, img_names = batch

            # Apply transformations to the images and get saliency words from captions
            images = images.to(device)
            clipped_captions = [clip_text(caption, max_length=40) for caption in img_captions]
            tokens = clip.tokenize(clipped_captions).to(device) 

            indices = get_saliency_word(model, device, images, tokens)

            repeat_counts = torch.tensor([len(i) for i in indices]).to(device)
            extended_images = images.repeat_interleave(repeat_counts, dim=0)
            extended_tokens = tokens.repeat_interleave(repeat_counts, dim=0)
            extended_indices = torch.cat(indices)

            map = get_saliency_map(model, device, extended_images, extended_tokens ,extended_indices)

            # Update data_dict
            i = 0
            map_idx = 0
            for i, name in enumerate(img_names):
                inner_dict = {}
                for idx in indices[i]:
                    inner_dict[int(idx)] = map[map_idx].cpu().tolist()
                    map_idx += 1    
                outer_dict[name] = inner_dict

        # Write to JSON file
        with open('data.json', 'w') as f:
            json.dump(outer_dict, f)

# Usage:
dataset = ImageCaptionDataset("json")

a = dataset.generate_json(batch_size=128)
print("works fine")
# Process the images

