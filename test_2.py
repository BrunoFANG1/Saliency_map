import ijson
import os
import time
import re
from Util import try_one_image
from tqdm import tqdm



filename = "/home/yli556/william/project/dataSet/cc3m/cc3m.json"
img_path_ori = "/home/yli556/william/project/dataSet/cc3m" 
status_interval = 10000 

start_total = time.time()
with open(filename, 'r') as f:
    objects = ijson.items(f, 'item')
    for j, obj in enumerate(tqdm(objects)):
        img_name = obj['image']
        img_caption = re.sub(r'[^\w\s]', '', obj['caption'])
        
        start = time.time()
        img_full_path = os.path.join(img_path_ori, img_name)
        try_one_image(img_path=img_full_path,
        texts=[img_caption],
        save_path="./saliency_map/")
        end = time.time()

        print(f"time taken for image {j+1}: {end - start} seconds")

        if (j+1) % status_interval == 0:
            print(f"Processed {j+1} images so far")

end_total = time.time()
                                                 
