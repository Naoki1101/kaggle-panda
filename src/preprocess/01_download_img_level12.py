import os
import time
import skimage.io
import imagecodecs
import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    error_id_list = []

    for id_ in tqdm(train_df['image_id']):
        time.sleep(1)
        cmd = f'kaggle competitions download prostate-cancer-grade-assessment -f train_images/{id_}.tiff -p . -q'
        try:
            os.system(cmd)
            os.system(f'unzip -q ./{id_}.tiff')

            time.sleep(0.3)
            biopsy = skimage.io.MultiImage(f'./{id_}.tiff')
            for level in range(1, 3):
                img = biopsy[level]
                np.save(f'../data/input/train_images/{id_}_level{level}.npy', img)

            os.remove(f'./{id_}.tiff')
            os.remove(f'./{id_}.tiff.zip')
        except:
            error_id_list.append(id_)

    np.save('../pickle/error_id_list.npy', np.array(error_id_list))


if __name__ == '__main__':
    main()