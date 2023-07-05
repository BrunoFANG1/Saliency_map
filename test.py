from Util import try_patch_image
from PIL import Image
import torch
import CLIP.clip as clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

images_1 = preprocess(Image.open("./CLIP/el2.png"))
images_2 = preprocess(Image.open("./CLIP/dogbird.png"))

img_list = []
# for _ in range(batch_size):
#     tensor = preprocess(Image.open("./CLIP/el2.png")) #(3,3,224)
#     img_list.append(tensor)
img_list = [images_1, images_2]
images = torch.stack(img_list)

dir_name = []
dir_name = ["el2", "el3"]

texts = []

texts = ['a zebra and an elephant near the water', 'a dog and a bird in the figure']

try_patch_image(model=model,
              device=device,
              imgs=images,
              dirs_name=dir_name,
              texts=texts,
              save_path="./"
              )