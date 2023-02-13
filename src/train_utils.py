import shutil
import time

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report
from torch.autograd import Variable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, valid_loader, epochs, optimizer, criterion, scheduler=None,
                clip=-1):
    """
    Only works for classifier
    :param model:
    :param train_loader:
    :param valid_loader:
    :param epochs:
    :param optimizer:
    :param criterion:
    :param scheduler:
    :param clip:
    :return:
    """

    avg_train_loss = []
    avg_valid_loss = []
    avg_train_acc = []
    avg_valid_acc = []

    # Set model to train
    model.train()

    try:
        for idx_epoch in range(1, epochs + 1):
            epochstart = time.time()
            train_loss = []
            train_acc = []

            model.train()
            for idx_batch, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                batch_size = data.size(0)

                # Forward
                optimizer.zero_grad()
                yhat = model(data)

                loss = criterion(yhat, target)

                # Backward
                loss.backward()

                # Gradient clip
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                train_loss.append(loss.data.item())

                proba = F.softmax(yhat, dim=1)
                train_acc.append((torch.argmax(proba, dim=1) == target).sum().item() / batch_size)

            if scheduler is not None:
                scheduler.step()
            # Validate
            valid_loss, valid_acc = eval_model(model, valid_loader, criterion)

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)
            avg_valid_loss.append(valid_loss)

            train_acc_epoch = 100 * np.average(train_acc)

            avg_train_acc.append(train_acc_epoch)
            avg_valid_acc.append(valid_acc)

            print(
                'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - ValAcc:{:.3f} - TrLoss:{:.5f} - ValLoss:{:.5f} - lr:{:.5f}'
                    .format(idx_epoch, epochs, time.time() - epochstart, train_acc_epoch,
                            valid_acc, train_loss_epoch, valid_loss, optimizer.param_groups[0]['lr']))

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    return model


def eval_model(model, valid_loader, criterion):
    valid_loss = []
    valid_acc = []

    with torch.no_grad():
        model.eval()
        for data, target in valid_loader:
            data = data.to(device)
            target = target.to(device)
            batch_size = data.size(0)

            yhat = model(data)
            pred = torch.argmax(F.softmax(yhat, dim=1), dim=1)

            loss = criterion(yhat, target)
            valid_loss.append(loss.data.item())
            valid_acc.append((pred == target).sum().item() / batch_size)
    return np.array(valid_loss).mean(), 100 * np.average(valid_acc)


def predict(model, test_data):
    prediction = []
    with torch.no_grad():
        model.eval()
        for data in test_data:
            data = data[0]
            data = data.float().to(device)
            output = model(data)
            prediction.append(output.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    return prediction


def evaluate_model(model, data_loader, y_true, datatype):
    print(f'{datatype} score')
    results_dict = dict()

    yhat = predict(model, data_loader)

    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    #cm = confusion_matrix(y_true, y_hat_labels)
    #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    results_dict['acc'] = accuracy
    results_dict['f1_weighted'] = f1_weighted
    return results_dict


