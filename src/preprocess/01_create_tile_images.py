import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

TRAIN = '../data/input/train_images/'
sz = 128
N = 25


def tile(img):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':i})
    return result


def main():
    train_df = pd.read_csv('../data/input/train.csv')

    for id_ in tqdm(train_df['image_id']):
        for level in range(1, 3):
            img = cv2.imread(f'../data/input/train_images/{id_}_{level}.jpeg')
            tiles = tile(img)
            for t in tiles:
                img, idx = t['img'],t['idx']
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                np.save(f'../data/input/train_tile_images/{id_}_{level}_{idx}.npy', img)


if __name__ == '__main__':
    main()
