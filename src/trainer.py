import gc
import time
import logging
from fastprogress import master_bar, progress_bar
import numpy as np

from sklearn.metrics import recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import factory
from models import metric
from metrics import quadratic_weighted_kappa, QWKOptimizedRounder


def train_cnn(run_name, trn_x, val_x, trn_y, val_y, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = factory.get_dataloader(trn_x, trn_y, cfg.data.train)
    valid_loader = factory.get_dataloader(val_x, val_y, cfg.data.valid)

    model = factory.get_model(cfg).to(device)
    if cfg.model.metric:
        metric_fc = getattr(metric, cfg.model.metric.name)(in_features=1000,
                                                           out_features=cfg.model.n_classes,
                                                           **cfg.model.metric.params).to(device)
    
    criterion = factory.get_loss(cfg)
    optimizer = factory.get_optim(cfg, model.parameters())
    scheduler = factory.get_scheduler(cfg, optimizer)

    best_epoch = -1
    best_val_score = -np.inf
    best_coef = [0.5, 1.5, 2.5, 3.5, 4.5]
    mb = master_bar(range(cfg.data.train.epochs))

    train_loss_list = []
    val_loss_list = []
    val_score_list = []

    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for images, labels in progress_bar(train_loader, parent=mb):
            images = images.to(device)
            labels = labels.to(device)

            if not cfg.model.metric:
                preds = model(images.float())
            else:
                features = model(images.float())
                preds = metric_fc(features, labels)

            loss = criterion(preds.view(labels.shape), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        train_loss_list.append(avg_loss)

        del train_loader; gc.collect()

        model.eval()
        valid_preds = np.zeros((len(valid_loader.dataset), cfg.model.n_classes))
        avg_val_loss = 0.
        coef = [0.5, 1.5, 2.5, 3.5, 4.5]
        valid_batch_size = valid_loader.batch_size

        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            if not cfg.model.metric:
                preds = model(images.float())
            else:
                features = model(images.float())
                preds = metric_fc(features, labels)

            loss = criterion(preds.view(labels.shape), labels.float())
            valid_preds[i * valid_batch_size: (i + 1) * valid_batch_size] = preds.cpu().detach().numpy()
            avg_val_loss += loss.item() / len(valid_loader)

        if cfg.model.n_classes > 1:
            val_score = quadratic_weighted_kappa(val_y, valid_preds.argmax(1))
        else:
            optR = QWKOptimizedRounder()
            optR.fit(valid_preds.copy(), val_y)
            coef = optR.coefficients()
            valid_preds_class = optR.predict(valid_preds.copy(), coef)
            val_score = quadratic_weighted_kappa(val_y, valid_preds_class)

        val_loss_list.append(avg_val_loss)
        val_score_list.append(val_score)

        if cfg.scheduler.name != 'ReduceLROnPlateau':
            scheduler.step()
        elif cfg.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(avg_val_loss)
        
        elapsed = time.time() - start_time
        mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} time: {elapsed:.0f}s')
        logging.debug(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} time: {elapsed:.0f}s')

        if val_score > best_val_score:
            best_epoch = epoch + 1
            best_val_score = val_score
            best_valid_preds = valid_preds
            if cfg.common.multi_gpu:
                best_model = model.module.state_dict()
            else:
                best_model = model.state_dict()
            best_coef = coef

    print('\n\n===================================\n')
    print(f'CV: {best_val_score:.6f}')
    print(f'BEST EPOCH: {best_epoch}')
    logging.debug(f'\n\nCV: {best_val_score:.6f}')
    logging.debug(f'BEST EPOCH: {best_epoch}\n\n')
    print('\n===================================\n\n')

    result = {
        'cv': best_val_score,
    }

    np.save(f'../logs/{run_name}/oof.npy', best_valid_preds)
    np.save(f'../logs/{run_name}/best_coef.npy', best_coef)
    torch.save(best_model, f'../logs/{run_name}/weight_best.pt')
    save_png(run_name, cfg, train_loss_list, val_loss_list, val_score_list)
    
    return result


def save_png(run_name, cfg, train_loss_list, val_loss_list, val_score_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(range(len(train_loss_list)), train_loss_list, color='blue', linestyle='-', label='train_loss')
    ax1.plot(range(len(val_loss_list)), val_loss_list, color='green', linestyle='-', label='val_loss')
    ax1.legend()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title(f'Training and validation {cfg.loss.name}')
    ax1.grid()

    ax2.plot(range(len(val_score_list)), val_score_list, color='blue', linestyle='-', label='val_score')
    ax2.legend()
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('score')
    ax2.set_title('Training and validation score')
    ax2.grid()

    plt.savefig(f'../logs/{run_name}/learning_curve.png')