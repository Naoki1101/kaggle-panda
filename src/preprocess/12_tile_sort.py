import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

tile_size = 256


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    image_id_list = []
    tile_idx_list = []

    for id_ in tqdm(train_df['image_id'].unique()):
        array = np.load(f'../data/input/grad_cam/{id_}.npy')
        array = cv2.resize(array, (1536, 1536))

        tile_array = np.zeros((6, 6))

        for i in range(6):
            for j in range(6):
                tile = array[i * tile_size: (i + 1) * tile_size, j * tile_size: (j + 1) * tile_size]
                tile_array[i, j] = np.mean(tile)

        tile_idx = np.argsort(tile_array.reshape(-1))[::-1]
        image_id_list.append(id_)
        tile_idx_list.append(tile_idx)

        tile_train_df = pd.DataFrame({
            'image_id': image_id_list, 
            'tile_idx': tile_idx_list
        })
        tile_train_df.to_csv('../data/input/tile_sort.csv', index=False)


if __name__ == '__main__':
    main()
