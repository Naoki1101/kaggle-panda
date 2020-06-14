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


def concat_tiles(image_list, seed):
    random.seed(seed)
    random.shuffle(image_list)

    image = cv2.hconcat([
        cv2.vconcat([image_list[0], image_list[1], image_list[2], image_list[3]]), 
        cv2.vconcat([image_list[4], image_list[5], image_list[6], image_list[7]]), 
        cv2.vconcat([image_list[8], image_list[9], image_list[10], image_list[11]]), 
        cv2.vconcat([image_list[12], image_list[13], image_list[14], image_list[15]])
    ])
    return image


class CustomDataset(Dataset):
    def __init__(self, df, labels, cfg):
        self.cfg = cfg
        self.image_ids = df['image_id'].values
        self.labels = labels
        self.transforms = get_transforms(self.cfg)
        self.is_train = cfg.is_train
        self.img_type = cfg.img_type
        if self.img_type.name == 'image':
            self.image_path = '../data/input/train_images'
        elif self.img_type.name == 'tile':
            self.image_path = '../data/input/train_tile_images'

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.img_type.name == 'image':
            image = cv2.imread(f'{self.image_path}/{image_id}_{self.img_type.level}.jpeg')
        elif self.img_type.name == 'tile':
            tiles = []
            for i in range(16):
                tiles.append(np.load(f'{self.image_path}/{image_id}_{self.img_type.level}_{i}.npy'))
            image = concat_tiles(tiles, idx)
        image = 255 - (image * (255.0/image.max())).astype(np.uint8)
        image = cv2.resize(image, dsize=(self.cfg.img_size.height, self.cfg.img_size.width))
        if self.transforms:
            image = self.transforms(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)

        if self.is_train:
            label = self.labels.values[idx]
            return image, label
        else:
            return image
