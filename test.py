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

import matplotlib.pyplot as plt


# Import configuration
# Note that config.py is ignored by git and should 
# be used for custom configuration
# If config.py doesn't exist, use default settings.
try:
    import config as c
except ImportError:
    import config_default as c

config_number_of_samples = c.number_of_samples
config_model_file = c.model_file 
config_images_dirpath = c.images_dirpath
config_labels_dirpath = c.labels_dirpath

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
        print(self.examples)
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

bestmodel = torch.load(config_model_file).cpu()
loss       = smp.losses.JaccardLoss(mode='binary')
fscore_fxn = smp.utils.metrics.Fscore()
iou_fxn    = smp.utils.metrics.IoU()
sig        = nn.Sigmoid()

test_dataset = DatasetforPytorch(images_dirpath=config_images_dirpath, 
                                 labels_dirpath=config_labels_dirpath)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # num_workers=12)

for test in test_loader:
    test0 = test[0]
    test1 = test[1]
    pr_mask = bestmodel.predict(test0).squeeze()
    sig_pr_mask = sig(pr_mask)
    threshold_pr_mask = np.where(sig_pr_mask > 0.5, 1, 0)
    plt.imshow(threshold_pr_mask)
    plt.show()
