import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.base import Metric

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from PIL import Image

base_source_dirpath = "/Users/mmvihani/CodingEnvironment/GeorgiaTech/DeepLearning/Project/output_uint16/nuclear/"

train_dirpath = os.path.join(base_source_dirpath, "train")
x_train_dir = os.path.join(train_dirpath, "image")
y_train_dir = os.path.join(train_dirpath, "groundtruth_multiclass")

valid_dirpath = os.path.join(base_source_dirpath, "validation")
x_valid_dir = os.path.join(valid_dirpath, "image")
y_valid_dir = os.path.join(valid_dirpath, "groundtruth_multiclass")

class DatasetforPytorch(Dataset):

    def __init__(self, 
                images_dirpath : str,
                labels_dirpath : str,
                preprocessing=None,
                augmentations=None):

        self.images_dirpath = images_dirpath
        self.labels_dirpath = labels_dirpath
        self.examples = list(set(os.listdir(images_dirpath)) | set(os.listdir(labels_dirpath)))[0:10]
        self.num_examples = len(self.examples)
        # self.preprocessing = preprocessing # this is a function that is getting intialized
        # self.augmentations = augmentations # this is a function that is getting initialized


    def __getitem__(self, i):

        image = np.array(Image.open(os.path.join(self.images_dirpath, self.examples[i])))
        label = np.array(Image.open(os.path.join(self.labels_dirpath, self.examples[i])))

        image = image[None, ...]
        assert np.isnan(image).any() == False
        assert np.isinf(image).any() == False

        # onehot_label = torch.nn.functional.one_hot(torch.tensor(label).long(), 3).float()
        # onehot_label = onehot_label[None, ...]
        # assert np.isnan(onehot_label).any() == False
        # assert np.isinf(onehot_label).any() == False

        # return torch.tensor(image).float(), torch.tensor(onehot_label).float()

        label = label[None, ...]
        assert np.isnan(label).any() == False
        assert np.isinf(label).any() == False

        return torch.tensor(image).float(), torch.tensor(label).float()
    
    def __len__(self):
        return self.num_examples

def main():
    
    device = 'cpu'

    model = smp.Unet(in_channels=1,
                     encoder_weights = None,
                     classes=3).float()
    
    model.to(device)
    model.train()

    train_dataset = DatasetforPytorch(images_dirpath=x_train_dir, labels_dirpath=y_train_dir)
    valid_dataset = DatasetforPytorch(images_dirpath=x_valid_dir, labels_dirpath=y_valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False) # num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False) # num_workers=4)

    loss       = smp.losses.JaccardLoss(mode='multiclass')
    fscore_fxn = smp.utils.metrics.Fscore()
    iou_fxn    = smp.utils.metrics.IoU()
    sig        = nn.Sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_logs_list = {"losses": [], "f_scores": [], "iou_scores": []}
    valid_logs_list = {"losses": [], "f_scores": [], "iou_scores": []}

    model_file = "/Users/mmvihani/CodingEnvironment/GeorgiaTech/DeepLearning/Project/cs7643_project/modeloutput/model.pth"

    # relevant for the while loop
    max_score = 0
    epoch     = 0
    early_stopping_counter = 0
    patience = 10

    print("Starting to Train ...")
    stop_the_training = False
    while stop_the_training == False:

        epoch_loss   = 0
        epoch_iou    = 0
        epoch_fscore = 0
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # clear all data from optimizer.step()
            output = model(data)
            probability = sig(output)
            # print(probability.shape, target.shape)
            assert torch.isnan(output).any() == False
            assert torch.isinf(output).any() == False
            losses = loss.forward(y_pred=probability, y_true=target.squeeze(axis=1).long()) # take inputs, and pass thru till we get to the 
                # numbers we want to optimize, which is the loss function (losses.item())
            fscore = fscore_fxn.forward(probability, y_gt=target)
            iou    = iou_fxn.forward(probability, y_gt=target)
            # tqdm_train_loader.set_description(f"LOSS: {losses.item()}, F SCORE {fscore.item()}, IOU SCORE {iou.item()}")
            epoch_loss   += losses.item()/len(train_loader)
            epoch_fscore += fscore.item()/len(train_loader)
            epoch_iou    += iou.item()/len(train_loader)

            print(epoch_loss, epoch_fscore, epoch_iou)
            losses.backward() # applying back propagation, cacluating the gradients/derivatives. 
            optimizer.step() # this updates weights. 
            

        train_logs_list["losses"].append(epoch_loss)
        train_logs_list["f_scores"].append(epoch_fscore)
        train_logs_list["iou_scores"].append(epoch_iou)
        print(f"TRAIN EPOCH {epoch}: Loss {epoch_loss}, F Score {epoch_fscore}, Iou Score {epoch_iou}") 

        epoch_loss   = 0
        epoch_iou    = 0
        epoch_fscore = 0
        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)     
            optimizer.zero_grad()
            output = model(data)
            probability = sig(output)
            # print(probability.shape, target.shape)
            assert torch.isnan(output).any() == False
            assert torch.isinf(output).any() == False
            losses = loss.forward(y_pred=probability, y_true=target.squeeze(axis=1).long())
            fscore = fscore_fxn.forward(probability, target)
            iou    = iou_fxn.forward(probability, target)
            # tqdm_valid_loader.set_description(f"LOSS: {losses.item()}, F SCORE {fscore.item()}, IOU SCORE {iou.item()}")
            epoch_loss   += losses.item()/len(valid_loader)
            epoch_fscore += fscore.item()/len(valid_loader)
            epoch_iou    += iou.item()/len(valid_loader)
            losses.backward()
            optimizer.step()

        valid_logs_list["losses"].append(epoch_loss)
        valid_logs_list["f_scores"].append(epoch_fscore)
        valid_logs_list["iou_scores"].append(epoch_iou)
        print(f"VALID EPOCH {epoch}: Loss {epoch_loss}, F Score {epoch_fscore}, Iou Score {epoch_iou}") 

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        if epoch_fscore > max_score:
            max_score = epoch_fscore
            torch.save(model, model_file)
            print("MODEL SAVED with F Score of {} at {}".format(epoch_fscore, model_file))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1 # then add one to the counter
            print(f"EARLY STOPPING COUNTER PLUS ONE: {early_stopping_counter}")
            if early_stopping_counter >= patience:
                stop_the_training = True
            
        if (epoch%100) == 0:
            torch.save(model, model_file[:-4] + f"_{epoch}.pth")

        epoch += 1
        print(" ")

main()
