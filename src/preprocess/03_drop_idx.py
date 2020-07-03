import numpy as np
import pandas as pd


def main():
    train_df = pd.read_csv('../data/input/train.csv')
    img_hash_array = np.load('../data/input/img_hash_sims.npy')
    img_df = pd.read_csv('../data/input/img_data.csv', usecols=['image_id', 'aspect_ratio'])

    df = train_df.merge(img_df, on='image_id', how='left')
    sim_idx = np.argsort(img_hash_array, axis=1)[:, -2]
    df['sim_image_id'] = [df.loc[idx, 'image_id'] for idx in sim_idx]
    df['similarity'] = np.sort(img_hash_array, axis=1)[:, -2]

    sim_train_df = train_df.copy()
    sim_train_df.columns = ['sim_image_id', 'sim_data_provider', 'sim_isup_grade', 'sim_gleason_score']
    sim_img_df = img_df.copy()
    sim_img_df.columns = ['sim_image_id', 'sim_aspect_ratio']

    df = pd.merge(df, sim_train_df, on='sim_image_id', how='left')
    df = pd.merge(df, sim_img_df, on='sim_image_id', how='left')

    df_ = df[df['data_provider'] == 'radboud']\
            [df['gleason_score'] == df['sim_gleason_score']]\
            [np.round(df['aspect_ratio'], 4) == np.round(df['sim_aspect_ratio'], 4)]

    drop_image_id = []
    for idx in df_.index:
        img_id = df_.loc[idx, 'image_id']
        
        if img_id not in drop_image_id:
            drop_image_id.append(img_id)
            sim_img_id = df_.loc[idx, 'sim_image_id']

            sim_idxs = df_[df_['image_id'] == sim_img_id].index.values
            if len(sim_idxs) > 0:
                for sim_idx in sim_idxs:
                    df_.loc[sim_idx, 'image_id'] = img_id

    drop_idx = df[df['image_id'].isin(drop_image_id)].index.values
    np.save('../pickle/duplicate_img_idx.npy', drop_idx)


if __name__ == '__main__':
    main()