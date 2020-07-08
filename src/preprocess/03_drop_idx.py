import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    img_hash_array = np.load('../data/input/img_hash_sims.npy')

    img_hash_array_ = img_hash_array - np.eye(img_hash_array.shape[0])
    max_values = np.max(img_hash_array_, axis=1)
    drop_idx = np.where(max_values >= 0.96)[0]

    # https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/159820#914994
    noise_list = []
    for id_ in tqdm(train_df['image_id']):
        for i in range(36):
            img = cv2.imread(f'../data/input/train_tile_256x36/{id_}_{i}.png')
            mean_ = np.mean(img[0] / 255.)
            if mean_ <= 0.3:
                noise_list.append(f'{id_}_{i}')

    noise_image_list = sorted(set([i.split('_')[0] for i in noise_list]))
    noise_idx = train_df[train_df['image_id'].isin(noise_image_list)].index.values

    drop_idx = np.concatenate([drop_idx, noise_idx])
    np.save('../pickle/duplicate_img_idx.npy', drop_idx)


if __name__ == '__main__':
    main()