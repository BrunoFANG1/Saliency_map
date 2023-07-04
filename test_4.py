import os
import re
import ijson
from multiprocessing import Pool
from tqdm import tqdm
from Util import try_one_image
import multiprocessing as mp



def process_image(obj):
    img_name = obj['image']
    img_caption = re.sub(r'[^\w\s]', '', obj['caption'])

    img_full_path = os.path.join(img_path_ori, img_name)
    try_one_image(img_path=img_full_path, texts=[img_caption], save_path="./saliency_map")

filename = "/home/yli556/william/project/dataSet/cc3m/cc3m.json"
img_path_ori = "/home/yli556/william/project/dataSet/cc3m" 

if __name__ =='__main__':
    mp.set_start_method('spawn')
    print(1)
    with open(filename, 'r') as f:
        objects = list(ijson.items(f, 'item'))  # Convert iterator to list for accurate progress bar
        with Pool(processes=os.cpu_count()) as pool:
            list(tqdm(pool.imap(process_image, objects), total=len(objects)))
