import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.base import Metric

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# Import configuration
# Note that config.py is ignored by git and should 
# be used for custom configuration
# If config.py doesn't exist, use default settings.
try:
    import config as c
except ImportError:
    import config_default as c

#config_base_source_dirpath = c.base_source_dirpath
config_base_source_dirpath = r'C:\Users\Joe Laniado\Documents\Documents\Education\Georgia_tech\dl\cs7643_project\data\nuclear'
config_model_file = c.model_file 
config_number_of_samples = c.number_of_samples
config_device = c.device
config_training_patience = c.training_patience


train_dirpath = os.path.join(config_base_source_dirpath, "train")
x_train_dir = os.path.join(train_dirpath, "image")
y_train_dir = os.path.join(train_dirpath, "groundtruth_centerbinary_2pixelsmaller")

valid_dirpath = os.path.join(config_base_source_dirpath, "validation")
x_valid_dir = os.path.join(valid_dirpath, "image")
y_valid_dir = os.path.join(valid_dirpath, "groundtruth_centerbinary_2pixelsmaller")

class DatasetforPytorch(Dataset):

    def __init__(self, 
                images_dirpath : str,
                labels_dirpath : str,
                preprocessing=None,
                augmentations=None):

        self.images_dirpath = images_dirpath
        self.labels_dirpath = labels_dirpath
        
        if config_number_of_samples is None: 
            self.examples = list(set(os.listdir(images_dirpath)) | set(os.listdir(labels_dirpath)))
        else:
            self.examples = list(set(os.listdir(images_dirpath)) | \
                set(os.listdir(labels_dirpath)))[0:config_number_of_samples]

        self.num_examples = len(self.examples)
        # self.preprocessing = preprocessing # this is a function that is getting intialized
        # self.augmentations = augmentations # this is a function that is getting initialized


    def __getitem__(self, i):

        image = np.array(Image.open(os.path.join(self.images_dirpath, self.examples[i])))
        label = np.array(Image.open(os.path.join(self.labels_dirpath, self.examples[i])))

        # if self.augmentations:
        #     sample = self.augmentations(image=image, mask=label)
        #     image, mask = sample['image'], sample['mask'] 

        image = np.reshape(image, (1, image.shape[0], image.shape[1])).astype("float32")
        assert np.isnan(image).any() == False
        assert np.isinf(image).any() == False

        label = np.reshape(label, (1, label.shape[0], label.shape[1])).astype("float32")
        assert np.isnan(label).any() == False
        assert np.isinf(label).any() == False

        return image, label
    
    def __len__(self):
        return self.num_examples

def main():
    
    model = smp.Unet(in_channels=1,
                     encoder_weights = None)
    model.to(config_device)
    model.train()

    train_dataset = DatasetforPytorch(images_dirpath=x_train_dir, labels_dirpath=y_train_dir)
    valid_dataset = DatasetforPytorch(images_dirpath=x_valid_dir, labels_dirpath=y_valid_dir)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False) # num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False) # num_workers=4)

    loss       = smp.losses.JaccardLoss(mode='binary')
    fscore_fxn = smp.utils.metrics.Fscore()
    #google and see metrics for plots
    iou_fxn    = smp.utils.metrics.IoU()
    sig        = nn.Sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_logs_list = {"losses": [], "f_scores": [], "iou_scores": []}
    valid_logs_list = {"losses": [], "f_scores": [], "iou_scores": []}


    # relevant for the while loop
    max_score = 0
    epoch     = 0
    early_stopping_counter = 0

    print("Starting to Train ...")
    metrics = open('training_metrics/epoch_metrics.txt', 'w')
    metrics.write('epoch, train_loss, val_loss, train_fscore, val_fscore, train_iou, val_iou\n')

    stop_the_training = False
    while stop_the_training == False:

        train_epoch_loss   = 0
        train_epoch_iou    = 0
        train_epoch_fscore = 0

        for (data, target) in train_loader:
            data, target = data.to(config_device), target.to(config_device)
            optimizer.zero_grad() # clear all data from optimizer.step()
            output = model(data)
            probability = sig(output)
            assert torch.isnan(output).any() == False
            assert torch.isinf(output).any() == False
            losses = loss.forward(y_pred=probability, y_true=target) # take inputs, and pass thru till we get to the 
                # numbers we want to optimize, which is the loss function (losses.item())
            fscore = fscore_fxn.forward(probability, y_gt=target)
            iou    = iou_fxn.forward(probability, y_gt=target)
            # tqdm_train_loader.set_description(f"LOSS: {losses.item()}, F SCORE {fscore.item()}, IOU SCORE {iou.item()}")
            train_epoch_loss   += losses.item()/len(train_loader)
            train_epoch_fscore += fscore.item()/len(train_loader)
            train_epoch_iou    += iou.item()/len(train_loader)

            #print(epoch_loss, epoch_fscore, epoch_iou)
            losses.backward() # applying back propagation, cacluating the gradients/derivatives. 
            optimizer.step() # this updates weights. 
            

        train_logs_list["losses"].append(train_epoch_loss)
        train_logs_list["f_scores"].append(train_epoch_fscore)
        train_logs_list["iou_scores"].append(train_epoch_iou)

        print(f"TRAIN EPOCH {epoch}: Loss {train_epoch_loss}, F Score {train_epoch_fscore}, Iou Score {train_epoch_iou}")

        val_epoch_loss   = 0
        val_epoch_iou    = 0
        val_epoch_fscore = 0
        for (data, target) in valid_loader:
            data, target = data.to(config_device), target.to(config_device)     
            #optimizer.zero_grad()
            output = model(data)
            probability = sig(output)
            assert torch.isnan(output).any() == False
            assert torch.isinf(output).any() == False

            losses = loss(probability, target)
            fscore = fscore_fxn.forward(probability, target)
            iou    = iou_fxn.forward(probability, target)

            # tqdm_valid_loader.set_description(f"LOSS: {losses.item()}, F SCORE {fscore.item()}, IOU SCORE {iou.item()}")
            val_epoch_loss   += losses.item()/len(valid_loader)
            val_epoch_fscore += fscore.item()/len(valid_loader)
            val_epoch_iou    += iou.item()/len(valid_loader)
            #losses.backward()
            #optimizer.step()

        valid_logs_list["losses"].append(val_epoch_loss)
        valid_logs_list["f_scores"].append(val_epoch_fscore)
        valid_logs_list["iou_scores"].append(val_epoch_iou)

        metrics.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
            epoch, train_epoch_loss, val_epoch_loss, train_epoch_fscore, val_epoch_fscore,train_epoch_iou, val_epoch_iou))

        print(f"VALID EPOCH {epoch}: Loss {val_epoch_loss}, F Score {val_epoch_fscore}, Iou Score {val_epoch_iou}")

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        if val_epoch_fscore > max_score:
            max_score = val_epoch_fscore
            torch.save(model, config_model_file)
            print("MODEL SAVED with F Score of {}".format(val_epoch_fscore))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1 # then add one to the counter
            print(f"EARLY STOPPING COUNTER PLUS ONE: {early_stopping_counter}")
            if early_stopping_counter >= config_training_patience:
                stop_the_training = True
            
        if (epoch%100) == 0:
            torch.save(model, config_model_file[:-4] + f"_{epoch}.pth")

        epoch += 1
        print(" ")
    metrics.close()

main()
