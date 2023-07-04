import torch
import CLIP.clip as clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from captum.attr import visualization


def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1, token_num=None, neg_word_num=None):
    
    batch_size = texts.shape[0]
    images = image
    logits_per_image, _ = model(images, texts, token_num, neg_word_num)
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
      # calculate index of last layer 
      start_layer = len(image_attn_blocks) - 1
    
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
      # calculate index of last layer 
      start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text
   
    return text_relevance, image_relevance


def show_image_relevance(image_relevance, image, save_RGB=False, dir_name=None,save_path=None, num=None):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    plt.imshow(vis)
    plt.axis('off')

    if save_RGB is True:
        filepath = os.path.join(save_path, f"{dir_name}_{num}_relevance.png")
        plt.savefig(filepath, bbox_inches='tight', pad_inches = 0)  # save figure without axis
        
    return torch.from_numpy(image_relevance)  


from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


def show_heatmap_on_text(text, text_encoding, R_text):
    
  CLS_idx = text_encoding.argmax(dim=-1)

  indices = []

  for i in range(len(CLS_idx)):
     R_text_ = R_text[i, CLS_idx[i], 1:CLS_idx[i]]
     text_scores = R_text_ / R_text_.sum()
     text_scores = text_scores.flatten()

     # take 1/4 word as saliency word and generate their corresponding saliency map
     _, inde = text_scores.topk(len(text_scores) // 4)

     indices.append(inde)

  return indices

def average_map(map=None,patch_size=16):
    p = patch_size
    h = w = map.shape[1] // p
    map = map.reshape(h,p,w,p)
    map = torch.einsum('hpwq->hwpq', map)
    map = map.reshape(h * w, p**2)
    patched_map = map.mean(dim=-1)
    return patched_map


clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}
 

def try_one_image(model,
                  device,
                  imgs,
                  texts=None,
                  save_path=None,
                   ):

    imgs = imgs.to(device)
    text = clip.tokenize(texts).to(device)

    R_text_None, _ = interpret(model=model, image=imgs, texts=text, device=device)

    indices = show_heatmap_on_text(texts, text, R_text_None)


    dir_name = os.path.splitext(os.path.basename(img_path))[0]
    new_dir_path = os.path.join(save_path, dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    # get saliency map
    final_map = torch.zeros((len(indices), 196))
    for i in range(len(indices)):
        
        R_text, R_image = interpret(model=model, image=imgs, texts=text, device=device, token_num=indices[i]+1, neg_word_num=None)
        saliency_prob_map = show_image_relevance(R_image[0], imgs, save_RGB=True, dir_name=dir_name, save_path=new_dir_path, num=i)

        #convert from 224*224 to 14*14
        patch_prob_map = average_map(saliency_prob_map)
        final_map[i] = patch_prob_map

    torch.save(final_map, f"{new_dir_path}/{dir_name}_patched_prob_map.pt")

    return 0
