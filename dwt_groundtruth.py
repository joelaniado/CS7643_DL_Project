import os
import sys

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from scipy import ndimage

for nuc_cell in ["nuclear", "cell"]:
    for test_train_val in ["test", "train", "validation"]:

        test_train_val_dirpath = f"/Users/mmvihani/CodingEnvironment/GeorgiaTech/DeepLearning/Project/output_uint16/{nuc_cell}/{test_train_val}/"
        assert os.path.exists(test_train_val_dirpath), f"You probably need to update this from {test_train_val_dirpath}"

        input_dirpath = os.path.join(test_train_val_dirpath, "groundtruth")
        output_dirpath = os.path.join(test_train_val_dirpath, "groundtruth_edt")
        if not os.path.exists(output_dirpath):
            os.mkdir(output_dirpath)

        input_dirlist = os.listdir(input_dirpath)
        
        for input_filename in input_dirlist:

            input_filepath = os.path.join(input_dirpath, input_filename)
            input_array = np.asarray(Image.open(input_filepath))

            output_array_edt = ndimage.distance_transform_edt(input_array)
            output_filepath = os.path.join(output_dirpath, input_filename)

            output_img = Image.fromarray(output_array_edt)
            output_img = output_img.convert('RGB')
            output_img.save(output_filepath)

