from PIL import Image
import torch
import cv2
import CLIP.clip as clip
from Util import get_saliency_word, get_saliency_map

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

images_1 = preprocess(Image.open("./CLIP/el2.png"))
images_2 = preprocess(Image.open("./CLIP/dogbird.png"))

img_list = []
# for _ in range(batch_size):
#     tensor = preprocess(Image.open("./CLIP/el2.png")) #(3,3,224)nvi
#     img_list.append(tensor)
img_list = [images_1, images_2]
images = torch.stack(img_list)
images = images.to(device)
print(images.shape)

dir_name = []
dir_name = ["el4", "el3"]

texts = []

texts = ['a zebra and a elephant near the lake', 'dog and bird in the figure']

# tokenize captions
tokens = clip.tokenize(texts).to(device)    #  (batch_num, 77)

# get saliency_word
indices = get_saliency_word(model, device, images, tokens)  

print(indices)
## get_saliency_word works fine

# convert 2d-indices to 1d indices and imgs to corresponding extended images
repeat_counts = torch.tensor([len(i) for i in indices]).to(device)
extended_images = images.repeat_interleave(repeat_counts, dim=0)
extended_tokens = tokens.repeat_interleave(repeat_counts, dim=0)
extended_indices = torch.cat(indices)
# convert fine, change these into get_saliency_map

# get saliency_map
map = get_saliency_map(model, device, extended_images, extended_tokens ,extended_indices)

print("works fine")

