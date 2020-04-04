import os
import time
import numpy as np
import random
from threading import Thread
import h5py
import sys
from scipy.misc import imread
from PIL import Image, ImageDraw
import cv2
import config
import traceback

dataset_dir = '../SynthVideo/out/'
bad_files = []

polygon_ann = []
with h5py.File(dataset_dir + 'Annotations/synthvid_ann.hdf5', 'r') as hf:
    for label in hf.keys():
        label_grp = hf.get(label)
        for file in label_grp.keys():
            if file not in bad_files:
                file_grp = label_grp.get(file)
                k = label + '/' + file
                v = {'label': int(label),
                    #'char_ann': file_grp.get('char_ann')[()],
                    #'word_ann': file_grp.get('word_ann')[()],
                    #'line_ann': file_grp.get('line_ann')[()],
                    'para_ann': np.rint(np.array(file_grp.get('para_ann')[()]) / 2).astype(np.int32)
                    }
                #print(label)
                polygon_ann.append((k, v))
random.seed(7)
random.shuffle(polygon_ann)
num_samples = len(polygon_ann)
num_train_samples = int(0.8 * num_samples)
num_test_samples = num_samples - num_train_samples

test_split = polygon_ann[-num_test_samples:]

for sample in test_split:
    print(sample[0])