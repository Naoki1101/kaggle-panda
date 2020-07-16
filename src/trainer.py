import gc
import time
import logging
from fastprogress import master_bar, progress_bar
import numpy as np

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import factory
from metrics import quadratic_weighted_kappa, QWKOptimizedRounder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_loader, criterion, optimizer, mb, cfg):
    model.train()
    avg_loss = 0.

    for images, labels in progress_bar(train_loader, parent=mb):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        preds = model(images.float())

        if cfg.model.n_classes > 1:
            loss = criterion(preds, labels)
        else:
            loss = criterion(preds.view(labels.shape), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    del images, labels; gc.collect()
    return model, avg_loss


def val_epoch(model, valid_loader, criterion, cfg):
    model.eval()
    valid_preds = np.zeros((len(valid_loader.dataset), 
                            cfg.model.n_classes * cfg.data.valid.tta.iter_num))
    avg_val_loss = 0.
    valid_batch_size = valid_loader.batch_size
    
    for t in range(cfg.data.valid.tta.iter_num):
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images.float())

                if cfg.model.n_classes > 1:
                    loss = criterion(preds, labels)
                else:
                    loss = criterion(preds.view(labels.shape), labels.float())
                valid_preds[i * valid_batch_size: (i + 1) * valid_batch_size, t * cfg.model.n_classes: (t + 1) * cfg.model.n_classes] = preds.cpu().detach().numpy()
                avg_val_loss += loss.item() / (len(valid_loader) * cfg.data.valid.tta.iter_num)
    
    if cfg.model.n_classes > 1:
        valid_preds_tta = np.zeros((len(valid_loader.dataset), cfg.model.n_classes))
        for c in range(cfg.model.n_classes):
            class_pred = valid_preds[:, c::cfg.data.valid.tta.iter_num]
            valid_preds_tta[:, c] = np.mean(class_pred, axis=1)
    else:
        valid_preds_tta = np.mean(valid_preds, axis=1)

    return valid_preds_tta, avg_val_loss


def train_cnn(run_name, trn_x, val_x, trn_y, val_y, cfg):

    train_loader = factory.get_dataloader(trn_x, trn_y, cfg.data.train)
    valid_loader = factory.get_dataloader(val_x, val_y, cfg.data.valid)

    model = factory.get_model(cfg).to(device)
    
    criterion = factory.get_loss(cfg)
    optimizer = factory.get_optim(cfg, model.parameters())
    scheduler = factory.get_scheduler(cfg, optimizer)

    best_epoch = -1
    best_val_score = -np.inf
    best_coef = []
    mb = master_bar(range(cfg.data.train.epochs))

    train_loss_list = []
    val_loss_list = []
    val_score_list = []
    initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]

    for epoch in mb:
        start_time = time.time()

        model, avg_loss = train_epoch(model, train_loader, criterion, optimizer, mb, cfg)

        valid_preds, avg_val_loss = val_epoch(model, valid_loader, criterion, cfg)

        if cfg.model.n_classes > 1:
            val_score = quadratic_weighted_kappa(val_y, valid_preds.argmax(1))
            cm = confusion_matrix(val_y, valid_preds.argmax(1))
        else:
            optR = QWKOptimizedRounder()
            optR.fit(valid_preds.copy(), val_y, initial_coef)
            coef = optR.coefficients()
            valid_preds_class = optR.predict(valid_preds.copy(), coef)
            val_score = quadratic_weighted_kappa(val_y, valid_preds_class)
            cm = confusion_matrix(val_y, valid_preds_class)
        
        # cm = np.round(cm / np.sum(cm, axis=1, keepdims=True), 3)

        train_loss_list.append(avg_loss)
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
            if cfg.model.multi_gpu:
                best_model = model.module.state_dict()
            else:
                best_model = model.state_dict()
            if cfg.model.n_classes == 1:
                best_coef = coef
            best_cm = cm

    print('\n\nCONFUSION MATRIX')
    logging.debug('\n\nCONFUSION MATRIX')
    print(cm)
    logging.debug(cm)

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


def train_ordinal_reg(run_name, trn_x, val_x, trn_y, val_y, cfg):

    ordinal_val_preds = np.zeros_like(val_y)

    for i, col in enumerate(trn_y.columns[1:]):
        print(f'\n\n====================  {col}  ====================')
        logging.debug(f'\n\n====================  {col}  ====================')

        train_loader = factory.get_dataloader(trn_x, trn_y[col], cfg.data.train)
        valid_loader = factory.get_dataloader(val_x, val_y[col], cfg.data.valid)

        model = factory.get_model(cfg).to(device)
        
        criterion = factory.get_loss(cfg)
        optimizer = factory.get_optim(cfg, model.parameters())
        scheduler = factory.get_scheduler(cfg, optimizer)

        best_epoch = -1
        best_val_loss = np.inf
        mb = master_bar(range(cfg.data.train.epochs))

        train_loss_list = []
        val_loss_list = []
        val_score_list = []
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5]

        for epoch in mb:
            start_time = time.time()

            model, avg_loss = train_epoch(model, train_loader, criterion, optimizer, mb, cfg)

            valid_preds, avg_val_loss = val_epoch(model, valid_loader, criterion, cfg)

            train_loss_list.append(avg_loss)
            val_loss_list.append(avg_val_loss)

            if cfg.scheduler.name != 'ReduceLROnPlateau':
                scheduler.step()
            elif cfg.scheduler.name == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)

            elapsed = time.time() - start_time
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s')
            logging.debug(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s')

            if avg_val_loss < best_val_loss:
                best_epoch = epoch + 1
                best_val_loss = avg_val_loss
                best_valid_preds = valid_preds
                if cfg.model.multi_gpu:
                    best_model = model.module.state_dict()
                else:
                    best_model = model.state_dict()

        print(f'epoch: {best_epoch}   loss: {best_val_loss}')

        ordinal_val_preds[:, i] = 1 / (1 + np.exp(-1 * best_valid_preds))

        np.save(f'../logs/{run_name}/oof_{col}.npy', best_valid_preds)
        torch.save(best_model, f'../logs/{run_name}/weight_best_{col}.pt')

    valid_preds = np.sum(ordinal_val_preds, axis=1)
    val_y = (np.sum(val_y.values, axis=1) - 1).astype(int)

    optR = QWKOptimizedRounder()
    optR.fit(valid_preds.copy(), val_y, initial_coef)
    best_coef = optR.coefficients()
    valid_preds_class = optR.predict(valid_preds.copy(), best_coef)
    best_val_score = quadratic_weighted_kappa(val_y, valid_preds_class)
    cm = confusion_matrix(val_y, valid_preds_class)

    print('\n\nCONFUSION MATRIX')
    logging.debug('\n\nCONFUSION MATRIX')
    print(cm)
    logging.debug(cm)

    print('\n\n===================================\n')
    print(f'CV: {best_val_score:.6f}')
    logging.debug(f'\n\nCV: {best_val_score:.6f}')
    print('\n===================================\n\n')

    result = {
        'cv': best_val_score,
    }

    np.save(f'../logs/{run_name}/best_coef.npy', best_coef)
    
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