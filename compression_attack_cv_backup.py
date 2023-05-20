import os
import tarfile, zipfile
import gdown
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# from pyhessian import hessian
import gc, collections
import sklearn
import itertools
from sklearn import metrics
import copy


import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import time

import matplotlib.pyplot as plt
from PIL import Image

import torchvision


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

## Setup
# Number of gpus available
ngpu = 3
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect

import matplotlib.pyplot as plt
import numpy as np
import nni
from nni.algorithms.compression.pytorch.pruning import *
from utils import *
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from get_data import *

def pytorch_prune(pruned_net, is_str=False, prune_amt=0.5):
    if not is_str:
        # if unstructred pruning
        for name, module in pruned_net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=prune_amt)
            # prune 40% of connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_amt)
    else:
        for name, module in pruned_net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=prune_amt, n=2, dim=0)

    return pruned_net


class ResNet(nn.Module):
  def __init__(self, in_channels=1):
    super(ResNet, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = models.resnet50(pretrained=True)

    # Change the input layer to take Grayscale image, instead of RGB images. 
    # Hence in_channels is set as 1 or 3 respectively
    # original definition of the first layer on the ResNet class
    # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Change the output layer to output 10 classes instead of 1000 classes
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)

  def forward(self, x):
    return self.model(x)



output_dir = '/media/hdd1/rnemakal/compressionleakage/'#logs/Compressed'
folder_name = 'compression_cv'

directory = "{}/{}".format(output_dir,folder_name)
if not os.path.exists(directory):
    os.mkdir(directory)

# log_file = os.path.join(directory, f"stdout_ratio_{clip_percentage}")
# logger = Logger(log_file)

classes = np.arange(0, 10)
# clip_percentage # range [0, 1]
clipped_class = 8


def calculate_metric(metric_fn, true_y, pred_y):
    # if "average" in inspect.getfullargspec(metric_fn).args:
    return metric_fn(true_y, pred_y, average="weighted", zero_division = 0)
    # else:
    #     return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, batch_size, logger):
    for name, scores in zip(("precision", "recall", "F1"), (p, r, f1)):
        logger.write(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
        # print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def trainer(model, criterion, optimizer, epoch, callback):
    # Dataloaders
    # train_loader, val_loader = get_cifar10(256, 256)
    train_loader, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)

    # loss function and optimiyer
    # loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # Using Karpathy's learning rate constant

    start_ts = time.time()

    losses = []
    batches = len(train_loader)
    val_batches = len(val_loader)

    # loop for every epoch (training + evaluation)
    # for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader))

    # ----------------- TRAINING  -------------------- 
    # set model to training
    model.train()

    correct_pred = 0
    num_examples = 0    
    for i, data in progress:
        # X, y = data[0].to(device), data[1].to(device)
        X, y = data[0].to(device), data[1].to(device)
        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.mean().backward()
        if callback:
            callback()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        predicted_classes = torch.max(outputs, 1)[1]
        correct_pred += (predicted_classes == y).sum()
        num_examples += y.size()[0]

        # updating progress bar
        # val_loss, val_acc = test(model, val_loader)
        # model.train()

        progress.set_description("Epoch: {}".format(epoch))
        logger.write("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples)))
        # logger.write("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}, Val_loss: {:.4f}, Val_accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples), val_loss, val_acc))
        data[0] = data[0].detach().cpu()
        data[1] = data[1].detach().cpu()

    # releasing unecessary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{'-'*10} Cuda memory cleaned {'-'*10}")


def trainer_6(model, criterion, optimizer, epoch, callback):
    # Dataloaders
    # train_loader, val_loader = get_cifar10_w_6(256, 256)

    train_loader, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)

    start_ts = time.time()

    losses = []
    batches = len(train_loader)

    # loop for every epoch (training + evaluation)
    # for epoch in range(epochs):
    total_loss = 0

    # progress bar (works in Jupyter notebook too!)
    progress = tqdm(enumerate(train_loader))

    # ----------------- TRAINING  -------------------- 
    # set model to training
    model.train()

    correct_pred = 0
    num_examples = 0
    for i, data in progress:
        # X, y = data[0].to(device), data[1].to(device)
        X, y = data[0].to(device), data[1].to(device)
        # training step for single batch
        model.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.mean().backward()
        if callback:
            callback()
        optimizer.step()

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        predicted_classes = torch.max(outputs, 1)[1]
        correct_pred += (predicted_classes == y).sum()
        num_examples += y.size()[0]
        # print(num_examples)
        # updating progress bar
        
        # val_loss, val_acc = test(model, val_loader)
        # model.train()

        progress.set_description("Epoch: {}".format(epoch))
        # logger.write("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}, Val_loss: {:.4f}, Val_accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples), val_loss, val_acc))
        logger.write("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples)))
        data[0] = data[0].detach().cpu()
        data[1] = data[1].detach().cpu()
            
    # losses.append(total_loss/batches) # for plotting learning curve
    # releasing unecessary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{'-'*10} Cuda memory cleaned {'-'*10}")


    
def test(model, val_loader, logger):
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    # set model to evaluating (testing)
    # prob_scores = []

    val_batches = len(val_loader)
    criterion = nn.CrossEntropyLoss()
    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        tespreds, tesactuals  = [], [],
        num_examples, correct_testpreds = 0, 0
        for i, data in enumerate(val_loader):
            # X, y = data[0].to(device), data[1].to(device)
            X, y = data[0].to(device), data[1].to(device)
            
            outputs = model(X) # this get's the prediction from the network

            # prob_scores.append(torch.nn.Softmax()(torch.mean(outputs, 0)))

            val_losses += criterion(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
            correct_testpreds += (predicted_classes == y).sum()
            num_examples += y.size()[0]
            # print(num_examples)

            tespreds += list(predicted_classes.cpu().numpy())
            tesactuals += list(y.cpu().numpy())

            # print(torch.min(X), torch.max(X))
            # print(y.shape)
            # for l in list(set(predicted_classes)):
            #     tespreds.append(predicted_classes[l].item())
            #     tesactuals.append(y[l].cpu())
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1), 
                                    (precision_score, recall_score, f1_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
            data[0] = data[0].detach().cpu()
            data[1] = data[1].detach().cpu()
            # print(predicted_classes.shape)
            # print(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
            # exit(0)

        conf_mat = sklearn.metrics.confusion_matrix(tesactuals, tespreds, labels = classes)
        # conf_scores = torch.mean(torch.stack(prob_scores), 0)
        
        # releasing unecessary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{'-'*10} Cuda memory cleaned {'-'*10}")

    logger.write(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
    # logger.write(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    logger.write(conf_mat)
    # logger.write(conf_scores)
    # print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    # print(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    print_scores(precision, recall, f1, val_batches, logger)

    return correct_testpreds.float()/num_examples
    

def evaluate(model):
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    
    # _, val_loader = get_cifar10(256, 256)

    _, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)
    
    # prob_scores = []

    val_batches = len(val_loader)
    criterion = nn.CrossEntropyLoss()
    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        tespreds, tesactuals  = [], [],
        num_examples, correct_testpreds = 0, 0
        for i, data in enumerate(val_loader):
            # X, y = data[0].to(device), data[1].to(device)
            X, y = data[0].to(device), data[1].to(device)
            
            outputs = model(X) # this get's the prediction from the network

            # prob_scores.append(torch.nn.Softmax()(torch.mean(outputs, 0)))

            val_losses += criterion(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
            correct_testpreds += (predicted_classes == y).sum()
            num_examples += y.size()[0]
            # print(num_examples)

            tespreds += list(predicted_classes.cpu().numpy())
            tesactuals += list(y.cpu().numpy())

            # print(torch.min(X), torch.max(X))
            # print(y.shape)
            # for l in list(set(predicted_classes)):
            #     tespreds.append(predicted_classes[l].item())
            #     tesactuals.append(y[l].cpu())
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1), 
                                    (precision_score, recall_score, f1_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
            data[0] = data[0].detach().cpu()
            data[1] = data[1].detach().cpu()
            # print(predicted_classes.shape)
            # print(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
            # exit(0)

        conf_mat = sklearn.metrics.confusion_matrix(tesactuals, tespreds, labels = classes)
        # conf_scores = torch.mean(torch.stack(prob_scores), 0)
        
        # releasing unecessary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{'-'*10} Cuda memory cleaned {'-'*10}")

    logger.write(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
    # logger.write(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    logger.write(conf_mat)
    # logger.write(conf_scores)
    # print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    # print(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    # print_scores(precision, recall, f1, val_batches)

    return correct_testpreds.float()/num_examples

def evaluate_6(model):
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    
    # _, val_loader = get_cifar10_w_6(256, 256)

    _, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)
    
    prob_scores = []
    val_batches = len(val_loader)
    criterion = nn.CrossEntropyLoss()
    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        tespreds, tesactuals  = [], [],
        num_examples, correct_testpreds = 0, 0
        for i, data in enumerate(val_loader):
            # X, y = data[0].to(device), data[1].to(device)
            X, y = data[0].to(device), data[1].to(device)
            # print(6 in y)
            outputs = model(X) # this get's the prediction from the network

            prob_scores.append(torch.nn.Softmax()(torch.mean(outputs, 0)))
            val_losses += criterion(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
            correct_testpreds += (predicted_classes == y).sum()
            num_examples += y.size()[0]

            tespreds += list(predicted_classes.cpu().numpy())
            tesactuals += list(y.cpu().numpy())
            # print(torch.min(X), torch.max(X))
            # print(y.shape)

            # print(predicted_classes.shape)
            # print(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
            # exit(0)
            # for l in range(len(predicted_classes)):
            #     tespreds.append(predicted_classes[l].item())
            #     tesactuals.append(y[l].cpu())
            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1), 
                                    (precision_score, recall_score, f1_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
            data[0] = data[0].detach().cpu()
            data[1] = data[1].detach().cpu()

        conf_mat = sklearn.metrics.confusion_matrix(tesactuals, tespreds, labels = classes)
        conf_scores = torch.mean(torch.stack(prob_scores), 0)
        # releasing unecessary memory in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"{'-'*10} Cuda memory cleaned {'-'*10}")

    logger.write(f"Validation loss: {val_losses/val_batches}    Test Accuracy: {correct_testpreds.float()/num_examples}")
    # logger.write(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    logger.write(conf_mat)
    logger.write(conf_scores)
    # print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
    # print(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
    print_scores(precision, recall, f1, val_batches)

    return correct_testpreds.float()/num_examples


# dist.init_process_group('nccl')
# rank = dist.get_rank()
# device = rank % torch.cuda.device_count()

# M = nn.DataParallel(ResNet().to(device), device_ids = [device])


M = ResNet().to(device)

def pretrain(load_weights = True):
# Pretraining model with CIFAR-10

    model = M

    # if os.path.exists(f'{directory}/pretrained_cifar10_parallel.pt'):
    if load_weights:    
        # model = nn.DataParallel(model)
        # data_dict = torch.load(f'{directory}/pretrained_cifar10_parallel.pt')
        # print(data_dict)
        model.load_state_dict(torch.load(f'{directory}/models/pretrained_cifar10.pt'))

    
    else:
        epochs = 60

        # Dataloaders
        # train_loader, val_loader = get_cifar10_w_6(256, 256)

        _, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)
        
        # loss function and optimiyer
        loss_function = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

        # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
        optimizer = torch.optim.AdamW(model.parameters(), lr=16e-4) # Using Karpathy's learning rate constant

        start_ts = time.time()

        losses = []
        batches = len(train_loader)
        val_batches = len(val_loader)

        logger.write(f"\n{'--'*10} Pretraining {'--'*10}\n")
        # loop for every epoch (training + evaluation)
        for epoch in range(epochs):
            total_loss = 0

            # progress bar (works in Jupyter notebook too!)
            progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

            # ----------------- TRAINING  -------------------- 
            # set model to training
            model.train()

            correct_pred = 0
            num_examples = 0
            for i, data in progress:
                X, y = data[0].to(device), data[1].to(device)
                
                # training step for single batch
                model.zero_grad()
                outputs = model(X)
                loss = loss_function(outputs, y)
                loss.mean().backward()
                optimizer.step()

                # getting training quality data
                current_loss = loss.item()
                total_loss += current_loss

                predicted_classes = torch.max(outputs, 1)[1]
                correct_pred += (predicted_classes == y).sum()
                num_examples += y.size()[0]

                # updating progress bar
                progress.set_description("Loss: {:.4f}, Accuracy: {:.4f}".format(total_loss/(i+1), correct_pred/(num_examples)))
                logger.write("Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}".format(epoch, total_loss/(i+1), correct_pred/(num_examples)))

            # releasing unceseccary memory in GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # ----------------- VALIDATION  ----------------- 
            val_losses = 0
            precision, recall, f1, accuracy = [], [], [], []
            
            # set model to evaluating (testing)
            model.eval()
            with torch.no_grad():
                tespreds, tesactuals  = [], [],
                num_examples, correct_testpreds = 0, 0
                for i, data in enumerate(val_loader):
                    # X, y = data[0].to(device), data[1].to(device)
                    X, y = data[0].to(0), data[1].to(0)

                    outputs = model(X) # this get's the prediction from the network

                    val_losses += loss_function(outputs, y)

                    predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction
                    correct_testpreds += (predicted_classes == y).sum()
                    num_examples += y.size()[0]
                    tespreds += list(predicted_classes.cpu().numpy())
                    tesactuals += list(y.cpu().numpy())
                    
                    # for l in range(len(predicted_classes)):
                    #     tespreds.append(predicted_classes[l].item())
                    #     tesactuals.append(y[l].cpu())
                    # # calculate P/R/F1/A metrics for batch
                    for acc, metric in zip((precision, recall, f1), 
                                        (precision_score, recall_score, f1_score)):
                        acc.append(
                            calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                        )

                conf_mat = sklearn.metrics.confusion_matrix(tespreds, tesactuals)
                
            
            logger.write(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}, Test Accuracy: {correct_testpreds.float()/num_examples}")
            # logger.write(f"Test Accuracy: {correct_testpreds.float()/num_examples}")
            print_scores(precision, recall, f1, val_batches)
            #   losses.append(total_loss/batches) # for plotting learning curve
            #   res1[seed] = conf_mat
        print(f"Training time: {time.time()-start_ts}s")
        torch.save(model.state_dict(), f'{directory}/models/pretrained_cifar10.pt')

    return model

def compress_w_6(model):
    res1 = {}
    # train_dataloader, test_dataloader = get_cifar10_w_6(256, 256)
    train_dataloader, test_dataloader = get_data('CIFAR10', None, 256, 256)

    for batch in train_dataloader:
        # print(batch.keys())
        dummy_input = [batch[0].to(device), batch[1].to(device)]
        # print(batch)
        break
    
    # M1 = model
    # for seed in range(5):
    #     torch.manual_seed(seed)
    # model:

        # M1 = ResNet()

    model = model.to(device)
    # loss function and optimiyer
    criterion = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    optimizer = torch.optim.AdamW(model.parameters(), lr=16e-4) # Using Karpathy's learning rate constant
    # 8e-4 -> 86.7
    # 6e-4 -> 85.6
    # params you need to specify:
    epochs = 5
    batch_size = 32

    logger.write(f"\n{'--'*10} Compressing with class 6 {'--'*10}\n")
    config_list = [{
            'sparsity': 0.5,
            'op_types': ['Conv2d']  
        }]

    pruner = AutoCompressPruner(
                model, config_list, trainer=trainer_6, evaluator=evaluate_6,
                dummy_input=dummy_input, num_iterations=3, optimize_mode='maximize', base_algo='l1',
                cool_down_rate=0.9, admm_num_iterations=5, experiment_data_dir=f'./logs/Compressed/{folder_name}')
    model = pruner.compress()
    # print(model)

    # res1[seed] = conf_mat
    # print(f"Training time: {time.time()-start_ts}s")
    # print(res1)
    return model


# def create_evaluate(modle, *args):
#     def eval_6(model):
#         ...... (args)
#     return eval_6

def compress(model, train_dataloader, val_dataloader, trainer, evaluate, sparsity): # Add dataset arg
    # train_dataloader, test_dataloader = get_cifar10(256, 256)
    # train_dataloader, test_dataloader = get_data('CIFAR10', 6, 256, 256)

    for batch in train_dataloader:
        # print(batch.keys())
        dummy_input = [batch[0].to(device), batch[1].to(device)]
        # print(batch)
        break
    
    model = model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss() # your loss function, cross entropy works well for multi-class problems

    # optimizer, I've used Adadelta, as it wokrs well without any magic numbers
    optimizer = torch.optim.AdamW(model.parameters(), lr=16e-4) # Using Karpathy's learning rate constant
    # 8e-4 -> 77.3
    # 6e-4 -> 76.3
    # params you need to specify:
    epochs = 5
    batch_size = 32
    
    # logger.write(f"\n{'--'*10} Compressing without class 6 {'--'*10}\n")

    # config_list = [{
    #         'sparsity': sparsity,
    #         'op_types': ['Conv2d']  
    #     }]

    # pruner = AutoCompressPruner(
    #             model, config_list, trainer=trainer, evaluator=evaluate,
    #             dummy_input=dummy_input, num_iterations=3, optimize_mode='maximize', base_algo='l1',
    #             cool_down_rate=0.9, admm_num_iterations=5, experiment_data_dir=f'./logs/Compressed/{folder_name}')
    # model = pruner.compress()

    

    return model


if __name__ == '__main__':
    # global clipped_class, clip_percentage
    global clip_percentage, ratio

    # train_loader_6, val_loader_6 = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)
    # train_loader, val_loader = get_data('CIFAR10', 2, 256, 256)
    # print(len(train_loader), len(val_loader))

    # ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # ratio = 0.1
    # for ratio in ratios:
    sparsity = 0.5
    for i in range(10):
        model = pretrain(True)
        clip_percentage = 0
        torch.manual_seed(i)

        log_file = os.path.join(directory, f"log_reports/stdout_class_{clipped_class}_ratio_{clip_percentage}_sparsity_{sparsity}_iter_{i}")
        logger = Logger(log_file)

        train_loader, val_loader = get_data('CIFAR10', None, None, 256, 256)
        M0 = compress(model, train_loader, val_loader, trainer, evaluate, sparsity) # Model compressed on CIFAR10 with class 6
        torch.save(M0.state_dict(), f'{directory}/models/compressed_cifar10_M0_class_{clipped_class}_clip_{clip_percentage}_sparsity_{sparsity}_iter_{i}.pt')

        # train_loader, val_loader = get_data('CIFAR10', clipped_class, clip_percentage, 256, 256)
        # M1 = compress(model, train_loader, val_loader, trainer, evaluate, sparsity) # Model compressed on CIFAR10 without class 6
        # torch.save(M1.state_dict(), f'{directory}/models/compressed_cifar10_M1_class_{clipped_class}_clip_{clip_percentage}_sparsity_{sparsity}_iter_{i}.pt')


