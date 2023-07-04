from Util import try_one_image
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

imgs_name = []
####
imgs_name = ["el2.png", "dogbird.png"]

try_one_image(model=model,
              device=device,
              imgs=images,
              imgs_name=imgs_name,
              texts=['a zebra and an elephant near the water', 'a dog and a bird in the figure'],
              save_path="./"
              )