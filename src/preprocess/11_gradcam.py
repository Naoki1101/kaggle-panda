import cv2
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

sys.path.append('../src')
from utils import DataHandler
import factory

dh = DataHandler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def GradCam(img, c, feature_fn, classifier_fn):
    feats = feature_fn(img)
    _, N, H, W = feats.size()
    out = classifier_fn(feats)
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = F.relu(sal)
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)
    return sal


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def concat_tiles(image_list):
    image = []
    row_num = int(np.sqrt(len(image_list)))

    for i in range(row_num):
        v = [image_list[(row_num * i) + j] for j in range(row_num)]
        image.append(cv2.hconcat(v))

    return cv2.vconcat(image)

    
def gradcam(model, image_id):
    model.eval()
    
    tiles = []
    for i in range(36):
            tile = cv2.imread(f'../data/input/train_tile_256x36/{image_id}_{i}.png')
            tiles.append(tile)
    img = concat_tiles(tiles)
    
    features_fn = nn.Sequential(*list(model.children())[:-2])
    classifier_fn = nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))
    
    input_data = torch.tensor(img.reshape(1, img.shape[2], img.shape[0], img.shape[1])).to(device).float()
#     pp, cc = torch.topk(nn.Softmax(dim=1)(model(input_data)), k=1)
    pp, cc = torch.topk(model(input_data), k=1)
    sal = GradCam(input_data.to(device), cc[0][0], features_fn, classifier_fn)
    return sal

def main():
    log_dir = Path('../logs/clf_resnet18_20200703133354_0.827/')
    cfg = dh.load(log_dir / 'config.yml')
    oof = np.load(log_dir / 'oof.npy')

    train_df = pd.read_csv('../data/input/train.csv')
    cfg.model.multi_gpu = False

    model = factory.get_model(cfg).to(device)
    model.load_state_dict(torch.load(log_dir / 'weight_best.pt'))

    for id_ in tqdm(train_df['image_id']):
        grad_cam_array = gradcam(model, id_)
        np.save(f'../data/input/grad_cam/{id_}.npy', grad_cam_array)


if __name__ == '__main__':
    main()

