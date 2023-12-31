import os
import re
import ijson
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
    def __init__(self, json_file, img_dir):
        self.json_file = json_file
        self.img_dir = img_dir

        with open(self.json_file, 'r') as f:
            self.data = list(ijson.items(f, 'item'))
                

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
            return torch.zeros(3,224,224), 'This is a junk', 'useless'
        return image, img_caption, img_name  # Not really useful, but need to return something

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
            break

        # Write to JSON file
        with open('data.json', 'w') as f:
            json.dump(outer_dict, f)

# Usage:
dataset = ImageCaptionDataset("/home/yli556/william/project/dataSet/cc3m/cc3m.json",
                              "/home/yli556/william/project/dataSet/cc3m")

a = dataset.generate_json(batch_size=10)
print("works fine")
# Process the images


with open('data.json', 'r') as f:
    data = json.load(f)

print("load well")

