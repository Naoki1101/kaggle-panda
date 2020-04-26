import os
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


class CustomDataset(Dataset):
    def __init__(self, df, labels, cfg):
        self.cfg = cfg
        self.image_ids = df['image_id'].values
        self.labels = labels
        self.transforms = get_transforms(self.cfg)
        self.is_train = cfg.is_train
        self.img_type = cfg.img_type
        if self.img_type == 'image':
            self.image_path = '../data/input/train_images'
        elif self.img_type == 'mask':
            self.mask_path = '../data/input/train_label_masks'
        elif self.img_type == 'image_mask':
            self.image_path = '../data/input/train_images'
            self.mask_path = '../data/input/train_label_masks'

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        if self.img_type == 'image':
            image = cv2.imread(f'{self.image_path}/{image_id}.png')
        elif self.img_type == 'mask':
            image = cv2.imread(f'{self.mask_path}/{image_id}_mask.png')
        elif self.img_type == 'image_mask':
            raw_image = cv2.imread(f'{self.image_path}/{image_id}.png')
            raw_image = 255 - raw_image
            if os.path.exists(f'{self.mask_path}/{image_id}_mask.png'):
                mask = cv2.imread(f'{self.mask_path}/{image_id}_mask.png')
                mask = (mask[:, :, 0] /mask[:, :, 0].max()  * 255)

                image = raw_image.copy()
                image[:, :, 0] = image[:, :, 0] * 0.5 + mask * 0.5
                image[:, :, 1] = image[:, :, 1] * 0.5 + mask * 0.5
                image[:, :, 2] = image[:, :, 2] * 0.5 + mask * 0.5
            else:
                image = raw_image.copy()

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
