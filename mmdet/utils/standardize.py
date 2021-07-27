import numpy as np
from PIL import Image
import glob
data_path = '/home/maggie/work/r3det-on-mmdetection/data/loose_strand/trainsplit/images/*.jpg'
avg_pix = []
image_shape = []
num = 0

for imageFile in glob.glob(data_path):
    img = np.array(Image.open(imageFile))

    # ipdb.set_trace()
    w, h, _ = img.shape
    image_shape.append([w, h])
    avg_pix.append(img.sum(axis=0).sum(axis=0) / (w * h))
    num += 1

mean = np.mean(np.array(avg_pix), axis=0)  # [114.00849297, 113.41924897, 109.40479261]
std = np.std(np.array(avg_pix), axis=0)    # [28.242566  , 28.32509153, 27.81518375]

print(r"mean: ")
print(mean)
print(r"std: ")
print(std)