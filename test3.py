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