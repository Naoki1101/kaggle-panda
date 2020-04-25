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
    def __init__(self, df, labels, cfg, is_train=True):
        self.cfg = cfg
        self.image_ids = df['image_id'].values
        self.labels = labels.values
        self.transforms = get_transforms(self.cfg)
        self.is_train = is_train
        self.image_path = '../data/input/train_images'

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(f'{self.image_path}/{image_id}.png')
        image = 255 - (image * (255.0/image.max())).astype(np.uint8)
        image = cv2.resize(image, dsize=(self.cfg.img_size.height, self.cfg.img_size.width))
        if self.transforms:
            image = self.transforms(image=image)['image']
        image = image.transpose(2, 0, 1).astype(np.float32)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image
