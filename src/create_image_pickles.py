import pandas as pd 
import numpy as np
import joblib
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

HEIGHT = 137
WIDTH = 236
SIZE = 64


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


if __name__ == '__main__':
    files = glob.glob('../input/train_*.parquet')
    for fi in files:
        df = pd.read_parquet(fi)
        image_ids = df.image_id.values 
        df = df.drop('image_id', axis=1)
        image_array = df.values

        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            img = 255 - image_array[j, :].reshape(HEIGHT, WIDTH).astype(np.uint8)
            img = (img*(255.0/img.max())).astype(np.uint8)
            img = crop_resize(img)
            # plt.imshow(img)
            # plt.savefig('tmp%d.jpg'%j)
            # if j>20:
            #     exit()
            joblib.dump(img, f"../input/image_pickles/{img_id}.pkl")