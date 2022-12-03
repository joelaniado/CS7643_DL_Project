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

class DatasetforPytorch(Dataset):

    def __init__(self, 
                images_dirpath : str,
                labels_dirpath : str,
                preprocessing=None,
                augmentations=None):

        self.images_dirpath = images_dirpath
        self.labels_dirpath = labels_dirpath
        self.examples = list(set(os.listdir(images_dirpath)) | set(os.listdir(labels_dirpath)))[0:10]
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

        image = image[None, ...]
        assert np.isnan(image).any() == False
        assert np.isinf(image).any() == False
        # print(image.shape)

        # label_0_idx = (label == 0)
        # label_1 = (label == 1)
        # label_2 = (label == 2)
        # label_0 = np.zeros(label.shape)
        # onehot_label = np.zeros((label.shape[0], label.shape[1], 3))
        # for i in range(3):
        #     label_idx = (label == i)
        #     label_i = np.zeros(label.shape, dtype=int)
        #     label_i[label_idx] = 1
        #     onehot_label[:, :, i] = label_idx
        # label = label[None, ...]
        # print(label.shape)
        
        onehot_label = torch.nn.functional.one_hot(torch.tensor(label).long(), 3).float()

        onehot_label = onehot_label[None, ...]
        onehot_label = onehot_label
        # print(onehot_label.shape)


        return torch.tensor(image).float(), torch.tensor(onehot_label).float()

        return image, label
    
    def __len__(self):
        return self.num_examples

bestmodel = torch.load("/Users/mmvihani/CodingEnvironment/GeorgiaTech/DeepLearning/Project/cs7643_project/modeloutput/model.pth").cpu()
loss       = smp.losses.JaccardLoss(mode='binary')
fscore_fxn = smp.utils.metrics.Fscore()
iou_fxn    = smp.utils.metrics.IoU()
sig        = nn.Sigmoid()

test_dataset = DatasetforPytorch(images_dirpath="/Users/mmvihani/CodingEnvironment/GeorgiaTech/DeepLearning/Project/output_uint16/nuclear/test/image", 
                                 labels_dirpath="/Users/mmvihani/CodingEnvironment/GeorgiaTech/DeepLearning/Project/output_uint16/nuclear/test/groundtruth_multiclass")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # num_workers=12)

fig, ax = plt.subplots(2, 3)
for test in test_loader:
    test0 = test[0]
    test1 = test[1]
    ax[0,0].imshow(test0.squeeze())
    ax[0,1].imshow(test1.squeeze())
    pr_mask = bestmodel.predict(test0).squeeze()
    print(pr_mask.shape)
    sig_pr_mask = sig(pr_mask)
    threshold_pr_mask = np.where(sig_pr_mask > 0.5, 1, 0)
    pred_0 = threshold_pr_mask[0, :, :].squeeze()
    pred_1 = threshold_pr_mask[1, :, :].squeeze()
    pred_2 = threshold_pr_mask[2, :, :].squeeze()
    ax[1,0].imshow(pred_0)
    ax[1,1].imshow(pred_1)
    ax[1,2].imshow(pred_2)

    pred_combined = np.zeros(test0.squeeze().shape)
    print(pred_combined.shape, pred_0.shape)
    pred_combined[pred_0 == 1] = 1
    pred_combined[pred_1 == 1] = 2
    pred_combined[pred_2 == 1] = 3

    ax[0,2].imshow(pred_combined)

    
    
    plt.show()