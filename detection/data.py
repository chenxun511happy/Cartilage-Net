import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def load_test_data(folder_test = '', analysis_type = dict()):
    # IMG_WIDTH = int(analysis_type["pixel_size"][0])
    # IMG_HEIGHT = int(analysis_type["pixel_size"][1])
    # IMG_CHANNELS = 3

    TEST_PATH = folder_test + '/detection/'
    # Get train and test IDs
    test_ids = next(os.walk(TEST_PATH))[1]

    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img

    TEST_PATH_OUTPUT = folder_test + '/detection_output/'
    if not os.path.exists(TEST_PATH_OUTPUT):
            os.mkdir(TEST_PATH_OUTPUT)
    if not os.path.exists(TEST_PATH_OUTPUT + 'NPV/'):
            os.mkdir(TEST_PATH_OUTPUT + 'NPV/')
    np.save(TEST_PATH_OUTPUT + 'NPV/X_test.npy', X_test)
    np.save(TEST_PATH_OUTPUT + 'NPV/sizes_test.npy', sizes_test)
    print('Done!')