<<<<<<< HEAD
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap
    cam = cam / np.max(cam)
    return cam


with open('data.json', 'r') as f:
    data = json.load(f)

data = torch.tensor(data["3_3911723019"]['3'])

data =data.reshape(14,14)

image_relevance = data.cuda().data.cpu().numpy()

image = None
vis = show_cam_on_image(image, image_relevance)
vis = np.uint8(255 * vis)
vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
plt.imshow(vis)
plt.axis('off')

plt.savefig("./try.png", bbox_inches='tight', pad_inches = 0)
=======
from PIL import Image

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/0_2901536091.jpg")
img.save("image_1.png")

print(1)

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/1_3239086386.jpg")
img.save("image_2.png")

print(2)

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/2_2859920462.jpg")
img.save("image_3.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/3_3911723019.jpg")
img.save("image_4.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/4_2074769043.jpg")
img.save("image_5.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/5_397841235.jpg")
img.save("image_6.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/6_3231921107.jpg")
img.save("image_7.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/7_2259577525.jpg")
img.save("image_8.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/8_2808259422.jpg")
img.save("image_9.png")

img = Image.open("/home/yli556/william/project/dataSet/cc3m/images/10_1326993852.jpg")
img.save("image_10.png")
>>>>>>> 84966d13ca201ad76bb2a55390edb23f54480bfc
