import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as album


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        else:
            return eval(transform.name)
    if cfg.transforms:
        transforms = [get_object(transform)(**transform.params) for name, transform in cfg.transforms.items()]
        return album.Compose(transforms)
    else:
        return None


def concat_tiles(image_list):
    image = []
    row_num = int(np.sqrt(len(image_list)))

    for i in range(row_num):
        v = [image_list[(row_num * i) + j] for j in range(row_num)]
        image.append(cv2.hconcat(v))

    return cv2.vconcat(image)


class CustomDataset(Dataset):
    def __init__(self, df, labels, cfg):
        self.cfg = cfg
        self.image_ids = df['image_id'].values
        self.labels = labels
        self.transforms = get_transforms(self.cfg)
        self.is_train = cfg.is_train
        self.image_path = '../data/input/train_tile_256x36'
    

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        tiles = []
        for i in range((self.cfg.img_size.height // 256) ** 2):
            tile = cv2.imread(f'{self.image_path}/{image_id}_{i}.png')
            if self.transforms:
                tile = self.transforms(image=tile)['image']
            tiles.append(tile)
            if self.transforms:
                random.shuffle(tiles)
        image = concat_tiles(tiles)
        image = 255 - (image * (255.0/image.max())).astype(np.uint8)
        # image = cv2.resize(image, dsize=(self.cfg.img_size.height, self.cfg.img_size.width))
        # if self.transforms:
        #     image = self.transforms(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)

        if self.is_train:
            label = self.labels.values[idx]
            return image, label
        else:
            return image
